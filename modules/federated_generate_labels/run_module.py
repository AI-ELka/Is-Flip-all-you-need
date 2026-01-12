"""
Optimizes logit labels given expert trajectories using trajectory matching.
"""

from pathlib import Path
import sys

import torch
import numpy as np

from modules.base_utils.datasets import get_matching_datasets, pick_poisoner, get_n_classes
from modules.base_utils.util import extract_toml, get_module_device, get_mtt_attack_info, \
                                    load_model, either_dataloader_dataset_to_both, make_pbar, \
                                    needs_big_ims, slurmify_path, clf_loss, softmax, total_mse_distance
from modules.federated_generate_labels.utils import coalesce_attack_config, extract_experts, \
                                                    extract_labels, sgd_step
from modules.base_utils.aggregator.trmean import aggr_trmean
from modules.base_utils.aggregator.krum import aggregate as aggr_krum


def cosine_similarity_list(grads_a, grads_b, eps=1e-8):
    dot, na, nb = 0.0, 0.0, 0.0
    for ga, gb in zip(grads_a, grads_b):
        dot += torch.sum(ga * gb)
        na += torch.sum(ga ** 2)
        nb += torch.sum(gb ** 2)
    return dot / (torch.sqrt(na) * torch.sqrt(nb) + eps)

def agg(params, grad_buf, method, f=1):
    agg_grads = []
    for i, p in enumerate(params):
        grads = torch.stack(grad_buf[i], dim=0)

        if method == "mean":
            g = grads.mean(dim=0)
        elif method == "median":
            g = grads.median(dim=0).values
        elif method == "trmean":
            g = aggr_trmean(grads, f=f)
        elif method == "krum":
            g = aggr_krum(grads, f=f)
        else:
            raise ValueError(method)
        p.grad = g
        agg_grads.append(g)
    return agg_grads


def run(experiment_name, module_name, **kwargs):
    """
    Optimizes and saves poisoned logit labels.

    :param experiment_name: Name of the experiment in configuration.
    :param module_name: Name of the module in configuration.
    :param kwargs: Additional arguments (such as slurm id).
    """

    slurm_id = kwargs.get('slurm_id', None)

    args = extract_toml(experiment_name, module_name)

    input_pths = args["input_pths"]
    opt_pths = args["opt_pths"]
    expert_model_flag = args["expert_model"]
    dataset_flag = args["dataset"]
    poisoner_flag = args["poisoner"]
    clean_label = args["source_label"]
    target_label = args["target_label"]
    lam = args.get("lambda", 0.0)
    train_pct = args.get("train_pct", 1.0)
    batch_size = args.get("batch_size", None)
    epochs = args.get("epochs", None)
    expert_config = args.get('expert_config', {})
    config = coalesce_attack_config(args.get("attack_config", {}))
    num_honests = args.get("num_honests", 2)
    num_poisoned = args.get("num_poisoned", 1)
    output_dir = slurmify_path(args["output_dir"], slurm_id)
    attack = args.get("attack", "backdoor")
    clean_trajectory = args.get("clean_trajectory", False)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Build datasets and initialize labels
    print("Building datasets...")
    poisoner = pick_poisoner(poisoner_flag,
                             dataset_flag,
                             target_label)

    big_ims = needs_big_ims(expert_model_flag)
    train_dataset, distill_dataset, test_dataset, poison_test_dataset, mtt_dataset =\
        get_matching_datasets(dataset_flag, poisoner, clean_label, train_pct=train_pct, big=big_ims, clean=clean_trajectory)
    
    n_classes = get_n_classes(dataset_flag)
    labels = extract_labels(mtt_dataset.distill, config['one_hot_temp'], n_classes)
    labels_init = torch.stack(extract_labels(mtt_dataset.distill, 1, n_classes))
    labels_syn = torch.stack(labels).requires_grad_(True)
    agg_method = args.get("agg_method", "mean")


    # Load expert trajectories
    print("Loading expert trajectories...")
    expert_starts, expert_opt_starts = extract_experts(
        expert_config,
        input_pths,
        config['iterations'],
        expert_opt_path=opt_pths
    )

    # Optimize labels
    print("Training...")

    student_model = load_model(expert_model_flag, n_classes)
    expert_model = load_model(expert_model_flag, n_classes)

    device = get_module_device(student_model)

    batch_size, epochs, optimizer_expert, optimizer_labels = get_mtt_attack_info(
        expert_model.parameters(),
        labels_syn,
        config['expert_kwargs'],
        config['labels_kwargs'],
        batch_size=batch_size,
        epochs=epochs
    )
    batch_size = batch_size // (num_honests + num_poisoned)
    loaders = []
    for _ in range(num_honests + num_poisoned):
        loader, _ = either_dataloader_dataset_to_both(
            mtt_dataset,
            batch_size=batch_size,
            shuffle=True
        )
        loaders.append(loader)

    losses = []

    with make_pbar(total=config['iterations'] * len(mtt_dataset)) as pbar:
        for it in range(config['iterations']):
            for batches in zip(*loaders):

                # Load expert trajectory
                checkpoint = torch.load(expert_starts[it])
                expert_model.load_state_dict(checkpoint)
                student_model.load_state_dict({k: v.clone() for k, v in checkpoint.items()})

                expert_start = [p.clone() for p in expert_model.parameters()]

                optimizer_expert.load_state_dict(torch.load(expert_opt_starts[it]))
                state_dict = torch.load(expert_opt_starts[it])

                expert_params = list(expert_model.parameters())
                student_params = list(student_model.parameters())

                expert_grad_buf = [[] for _ in expert_params]
                student_grad_buf = [[] for _ in student_params]

                # Compute gradients
                for cid, batch in enumerate(batches):

                    # ---------- HONEST CLIENTS ----------
                    if cid < num_honests:
                        x, y = batch[0].to(device), batch[1].to(device)

                        # Expert
                        expert_model.zero_grad()
                        loss_e = clf_loss(expert_model(x), y)
                        loss_e.backward()

                        for i, p in enumerate(expert_params):
                            if p.grad is not None:
                                expert_grad_buf[i].append(p.grad.detach().clone())

                        # Student
                        loss_s = clf_loss(student_model(x), y)
                        grads_s = torch.autograd.grad(
                            loss_s, student_params, create_graph=True
                        )

                        for i, g in enumerate(grads_s):
                            student_grad_buf[i].append(g)

                    # ---------- POISONED CLIENTS ----------
                    else:
                        x_t, y_t, x_d, _, idx = batch
                        x_t, y_t = x_t.to(device), y_t.to(device)
                        x_d = x_d.to(device)
                        y_d = labels_syn[idx].to(device)

                        # Expert
                        expert_model.zero_grad()
                        loss_e = clf_loss(expert_model(x_t), y_t)
                        loss_e.backward()

                        for i, p in enumerate(expert_params):
                            if p.grad is not None:
                                expert_grad_buf[i].append(p.grad.detach().clone())

                        # Student
                        loss_s = clf_loss(student_model(x_d), softmax(y_d))
                        grads_s = torch.autograd.grad(
                            loss_s, student_params, create_graph=True
                        )

                        for i, g in enumerate(grads_s):
                            student_grad_buf[i].append(g)

                # Aggregate expert gradients
                agg_expert_grads = agg(
                    expert_params,
                    expert_grad_buf,
                    agg_method, 
                    f=num_poisoned
                )

                optimizer_expert.step()
                expert_model.eval()

                # Aggregate student gradients (DIFFERENTIABLE)
                agg_student_grads = agg(
                    student_params,
                    student_grad_buf,
                    agg_method, 
                    f=num_poisoned
                )

                # MTT objective
                param_loss = torch.tensor(0.0, device=device)
                param_dist = torch.tensor(0.0, device=device)

                reg_term = lam * torch.linalg.vector_norm(
                    softmax(labels_syn) - labels_init,
                    ord=1,
                    dim=1
                ).mean()

                if attack in ["backdoor", "untargeted"]:
                    for init, student, expert, grad, state in zip(
                        expert_start,
                        student_params,
                        expert_params,
                        agg_student_grads,
                        state_dict["state"].values(),
                    ):
                        student_update = sgd_step(
                            student, grad, state, state_dict["param_groups"][0]
                        )

                        param_loss += total_mse_distance(student_update, expert)
                        param_dist += total_mse_distance(init, expert)

                    grand_loss = (param_loss / param_dist) + reg_term
                    if attack == "untargeted":
                        grand_loss = -grand_loss
                
                elif attack == "grad_ascent":
                    cos = cosine_similarity_list(
                        agg_student_grads,
                        agg_expert_grads
                    )
                    grand_loss = cos + reg_term

                elif attack == "orth_grad":
                    cos = cosine_similarity_list(
                        agg_student_grads,
                        agg_expert_grads
                    )
                    grand_loss = cos ** 2 + reg_term

                # Optimize labels
                optimizer_labels.zero_grad()
                grand_loss.backward()
                optimizer_labels.step()

                losses.append(grand_loss.item())
                pbar.update(batch_size)
                pbar.set_postfix(
                    g_loss=f"{np.mean(losses[-20:]):.4g}",
                    reg=f"{reg_term.item():.4g}"
                )

    # Save results
    print("Saving results...")
    y_true = torch.stack([mtt_dataset[i][3].detach() for i in range(len(mtt_dataset.distill))])
    np.save(output_dir + "labels.npy", labels_syn.detach().numpy())
    np.save(output_dir + "true.npy", y_true)
    np.save(output_dir + "losses.npy", losses)

if __name__ == "__main__":
    experiment_name, module_name = sys.argv[1], sys.argv[2]
    run(experiment_name, module_name)
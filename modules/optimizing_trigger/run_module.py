from modules.base_utils.datasets import get_n_classes, pick_poisoner
from modules.base_utils.util import get_train_info, mini_train, load_model,either_dataloader_dataset_to_both, extract_toml, slurmify_path, make_pbar, need_big_ims
from modules.optimizing_trigger.utils import sample_checkpoints, cosine_grad_loss, compute_batch_gradients, trigger_penalty, get_mu, extract_experts, get_clean_dataset, get_poison_dataset, move_to_device, init_delta, raw_to_preprocess, raw_to_trigger_preprocess, get_raw_clean_dataset, match_loss
from modules.train_expert.utils import checkpoint_callback
import torch
from pathlib import Path
import os
import matplotlib.pyplot as plt
import numpy as np
import copy
import torchvision.transforms as transforms
from pathlib import Path
import copy
import torch
import numpy as np
import matplotlib.pyplot as plt

def optimize_trigger_step(
    expert_models,
    K,
    raw_train_loader,
    source_label,
    target_label,
    loss_fn,
    delta,
    mu,
    optimizer_delta,
    lambda_match,
    lambda_adv,
    lambda_penalty,
    lambda_delta,
    alpha_ckpt,
    num_chckpt,
    epsilon,
    device="cuda",
):
    sampled_k = sample_checkpoints(K, num_chckpt, alpha=alpha_ckpt, device=device)

    pbar = make_pbar(raw_train_loader, desc=f"Optimizing trigger", leave=False)

    for batch in pbar:
        x_raw, y = move_to_device(batch, device)
        x_clean = raw_to_preprocess(x_raw)

        mask = (y == source_label)
        y_poison = y.clone()
        y_poison[mask] = target_label

        x_poisoned = x_clean.clone()
        x_poisoned[mask] = raw_to_trigger_preprocess(x_raw[mask], delta)

        optimizer_delta.zero_grad()

        clean_grad_sum = None
        poison_grad_sum = None
        logits_adv_sum = None

        for k in sampled_k:
            M = expert_models[k].to(device).eval()

            grads_clean, _ = compute_batch_gradients(
                M, loss_fn, (x_clean, y),
                create_graph=False, retain_graph=False
            )
            g_clean = torch.cat([g.view(-1) for g in grads_clean]).detach()

            grads_poison, _ = compute_batch_gradients(
                M, loss_fn, (x_poisoned, y_poison),
                create_graph=True, retain_graph=True
            )
            g_poison = torch.cat([g.view(-1) for g in grads_poison])

            if clean_grad_sum is None:
                clean_grad_sum = g_clean
                poison_grad_sum = g_poison
                logits_adv_sum = M(x_poisoned)
            else:
                clean_grad_sum += g_clean
                poison_grad_sum += g_poison
                logits_adv_sum += M(x_poisoned)

            # M.to("cpu")
            # torch.cuda.empty_cache()

        clean_grad = clean_grad_sum / len(sampled_k)
        poison_grad = poison_grad_sum / len(sampled_k)
        logits_adv = logits_adv_sum / len(sampled_k)

        L_match = cosine_grad_loss(clean_grad, poison_grad)
        L_adv = loss_fn(logits_adv, y_poison)
        L_pen = trigger_penalty(delta, mu)

        L_tot = (
            lambda_match * L_match
            + lambda_adv * L_adv
            + lambda_penalty * L_pen
            + lambda_delta * delta.norm()
        )

        L_tot.backward()
        optimizer_delta.step()

        with torch.no_grad():
            delta.clamp_(-epsilon, epsilon)

        pbar.set_postfix({
                 "L_match": f"{L_match.item():.4f}",
                 "L_adv": f"{L_adv.item():.4f}",
                 "L_pen": f"{L_pen.item():.4f}",
                 "||delta||": f"{delta.norm().item():.4f}"
             })
        delta_img = delta.detach().cpu().numpy().transpose(1, 2, 0)
    delta_img = (delta_img - delta_img.min()) / (delta_img.max() - delta_img.min() + 1e-8)
    plt.imshow(delta_img)
    plt.title("Optimized Trigger (Delta)")
    plt.axis("off")
    plt.savefig(f"out/optimizing_trigger_stripe/opt_trig_FLIP.png")

    return delta

def optimize_trigger(
    model,
    loss_fn,
    dataset_flag,
    mu,
    source_label,
    target_label,
    lambda_match=1.0,
    lambda_adv=1.0,
    lambda_penalty=0.1,
    lambda_delta=0.01,
    alpha_ckpt=0.1,
    num_chckpt=4,
    epsilon=0.03,
    lr_delta=1e-2,
    n_steps=100,
    device="cuda",
    train_flag="sgd",
    batch_size=None,
    optim_kwargs=None,
    scheduler_kwargs=None,
    expert_config=None,
    expert_path="/Data/mb/flip/out/checkpoints/r32p_1xs/{}/model_{}_{}.pth",
    chkpt_iters=50,
    output_dir="/Data/mb/flip/out/checkpoints/r32p_1xs/0/",
    epochs=20,
    init="stripe",
):

    optim_kwargs = optim_kwargs or {}
    scheduler_kwargs = scheduler_kwargs or {}
    expert_config = expert_config or {}

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    delta = init_delta(
        mu.shape,
        horizontal=True,
        strength=6.0,
        freq=16,
        device=device,
        init=init
    )
    delta.requires_grad_(True)

    optimizer_delta = torch.optim.Adam([delta], lr=lr_delta)

    raw_train_dataset = get_raw_clean_dataset(dataset_flag, train=True)
    raw_train_loader, _ = either_dataloader_dataset_to_both(
        raw_train_dataset,
        batch_size=256,
        shuffle=True,
    )

    checkpoints_start = extract_experts(expert_config, expert_path)

    big_ims = need_big_ims(dataset_flag)

    for step in range(n_steps):
        print(f"\n=== Trigger optimization step {step+1}/{n_steps} ===")

        delta_eval = delta.clone().detach().cpu()

        batch_size_, epochs_, opt, lr_scheduler = get_train_info(
            model.parameters(),
            train_flag,
            batch_size=batch_size,
            epochs=epochs,
            optim_kwargs=optim_kwargs,
            scheduler_kwargs=scheduler_kwargs,
        )

        poison_train_dataset = get_poison_dataset(
            dataset_flag=dataset_flag,
            source_label=source_label,
            target_label=target_label,
            delta=delta_eval,
            train=True,
            big=big_ims,
        )

        clean_test_dataset = get_clean_dataset(
            dataset_flag=dataset_flag, train=False, big=big_ims
        )

        poison_test_dataset = get_poison_dataset(
            dataset_flag=dataset_flag,
            source_label=source_label,
            target_label=target_label,
            delta=delta_eval,
            train=False,
            big=big_ims,
        )

        mini_train(
            model=model,
            train_data=poison_train_dataset,
            test_data=[clean_test_dataset, poison_test_dataset],
            batch_size=batch_size_,
            opt=opt,
            scheduler=lr_scheduler,
            epochs=epochs_,
            callback=lambda m, o, e, i: checkpoint_callback(
                m, o, e, i, chkpt_iters, output_dir
            ),
        )

        expert_models = []
        for ckpt_path in checkpoints_start:
            M = copy.deepcopy(model).to(device)
            state = torch.load(ckpt_path, map_location=device)
            M.load_state_dict(state)
            M.eval()
            expert_models.append(M)

        delta = optimize_trigger_step(
            expert_models=expert_models,
            K=len(expert_models),
            raw_train_loader=raw_train_loader,
            source_label=source_label,
            target_label=target_label,
            loss_fn=loss_fn,
            delta=delta,
            mu=mu,
            optimizer_delta=optimizer_delta,
            lambda_match=lambda_match,
            lambda_adv=lambda_adv,
            lambda_penalty=lambda_penalty,
            lambda_delta=lambda_delta,
            alpha_ckpt=alpha_ckpt,
            num_chckpt=num_chckpt,
            epsilon=epsilon,
            device=device,
        )

        del expert_models
        torch.cuda.empty_cache()

        # torch.save(
        #     delta.detach().cpu(),
        #     Path(output_dir) / f"out/optimizing_trigger/optimized_trigger_step_{step+1}.pt",
        # )

    return delta.detach()

def run(experiment_name, module_name, **kwargs):
    """
    Optimizes and saves a trigger (delta, rho).
    """
    slurm_id = kwargs.get("slurm_id", None)

    args = extract_toml(experiment_name, module_name)

    dataset_flag = args["dataset"]
    model_flag = args["model"]
    y_source = args["source_label"]
    y_target = args["target_label"]

    lambda_match = args.get("lambda_match", 0.0)
    lambda_adv = args.get("lambda_adv", 0.0)
    lambda_penalty = args.get("lambda_penalty", 0.0)
    lambda_rho = args.get("lambda_rho", 0.0)
    lambda_delta = args.get("lambda_delta", 0.0)

    epsilon = args.get("epsilon", 0.1)
    lr_delta = args.get("lr_delta", 1e-2)
    n_steps = args.get("n_steps", 100)

    alpha_ckpt = args.get("alpha_ckpt", None)
    num_chckpt = args.get("num_chckpt", 15)
    
    expert_config = args.get("expert_config", {})
    expert_path = args.get("expert_path", None)

    device = args.get("device", "cuda" if torch.cuda.is_available() else "cpu")

    init = args.get("init", "stripe")

    output_dir = slurmify_path(args["output_dir"], slurm_id)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    n_classes = get_n_classes(dataset_flag)
    model = load_model(model_flag, n_classes).to(device)

    print("Model loaded on device:", device)

    model.eval()

    mu = get_mu(dataset_flag, y_target, device)

    print("Optimizing trigger...")
    loss_fn = torch.nn.CrossEntropyLoss()
    # , optimized_rho_logit
    optimized_delta = optimize_trigger(
        model=model,
        dataset_flag=dataset_flag,
        loss_fn=loss_fn,
        mu=mu,
        source_label=y_source,
        target_label=y_target,
        lambda_match=lambda_match,
        lambda_adv=lambda_adv,
        lambda_penalty=lambda_penalty,
        lambda_delta=lambda_delta,
        alpha_ckpt=alpha_ckpt,
        num_chckpt=num_chckpt,
        epsilon=epsilon,
        lr_delta=lr_delta,
        n_steps=n_steps,
        device=device,
        expert_config=expert_config,
        expert_path=expert_path,
        init=init
    )

    print("Optimized trigger obtained.")

    torch.save(
        optimized_delta.cpu(),
        f"out/optimizing_trigger/optimized_trigger_final.pt",
    )
    # torch.save(
    #     optimized_rho_logit.cpu(),
    #     Path(output_dir) / "optimized_rho_logit.pt",
    # )

if __name__ == "__main__":
    run("optimizing_trigger_example", "optimizing_trigger")
# import torch
# import random
# import math
# import numpy as np
# import torch.nn.functional as F
# from modules.base_utils.datasets import get_n_classes, load_dataset, MappedDataset, TRANSFORM_TRAIN_XY, TRANSFORM_TEST_XY, MappedDataset, pick_poisoner, CIFAR_TRANSFORM_NORMALIZE_MEAN, CIFAR_TRANSFORM_NORMALIZE_STD, CIFAR_100_TRANSFORM_NORMALIZE_MEAN, CIFAR_100_TRANSFORM_NORMALIZE_STD, SVHN_TRANSFORM_NORMALIZE_MEAN, SVHN_TRANSFORM_NORMALIZE_STD, TINY_IMAGENET_TRANSFORM_NORMALIZE_MEAN, TINY_IMAGENET_TRANSFORM_NORMALIZE_STD
# from modules.federated_generate_labels.utils import DEFAULT_EXPERT_CONFIG, DEFAULT_ATTACK_ITERATIONS
# from modules.base_utils.util import need_big_ims
# from PIL import Image
# from torchvision import transforms
# from torch.utils.data import Subset, ConcatDataset
# import numpy as np

# RAW_TRANSFORM_X = transforms.ToTensor()
# RAW_TRANSFORM_Y = lambda y: y
# # To complete with size 

# DATASET_NORMALIZATION = {
#     'cifar': (CIFAR_TRANSFORM_NORMALIZE_MEAN, CIFAR_TRANSFORM_NORMALIZE_STD),
#     'cifar_100': (CIFAR_100_TRANSFORM_NORMALIZE_MEAN, CIFAR_100_TRANSFORM_NORMALIZE_STD),
#     'svhn': (SVHN_TRANSFORM_NORMALIZE_MEAN, SVHN_TRANSFORM_NORMALIZE_STD),
#     'tiny_imagenet': (TINY_IMAGENET_TRANSFORM_NORMALIZE_MEAN, TINY_IMAGENET_TRANSFORM_NORMALIZE_STD),
# }

# def get_transforms(dataset_flag, train=True, big=False, raw=False):
#     if raw:
#         return RAW_TRANSFORM_X, RAW_TRANSFORM_Y
#     if train:
#         transform = TRANSFORM_TRAIN_XY[dataset_flag + ('_big' if big else '')]
#     else:
#         transform = TRANSFORM_TEST_XY[dataset_flag + ('_big' if big else '')]
#     return transform

# def sample_checkpoints(K, S, alpha=0.01, device="cpu"):
#     ks = torch.arange(0, K, device=device, dtype=torch.float)
#     probs = torch.exp(-alpha * ks)
#     probs = probs / probs.sum()
#     idx = torch.multinomial(probs, S, replacement=False)
#     idx = idx.tolist()
#     if K-1 not in idx:
#         idx.append(K-1)
#     idx.sort()
#     return idx

# def init_delta(mu_shape, strength=6.0, freq=16, horizontal=True, device="cuda", init='stripe'):
#     if init == 'stripe':
#         C, H, W = mu_shape
#         sin_1d = torch.sin(torch.linspace(0, freq * torch.pi, W, device=device))
#         mask = sin_1d.view(1, 1, W).expand(C, H, W)
#         if horizontal:
#             mask = mask.transpose(1, 2)
#         delta = strength * mask
#         delta.requires_grad_(True)
#     elif init == 'random':
#         delta = strength * torch.rand(mu_shape, device=device)
#         delta.requires_grad_(True)
#     else:
#         raise ValueError(f"Unknown init type: {init}")
#     return delta

# def cosine_grad_loss(grads_clean, grads_poison, eps=1e-8):
#     loss = F.cosine_similarity(torch.cat([g.view(-1) for g in grads_clean if g is not None]),torch.cat([g.view(-1) for g in grads_poison if g is not None]), dim=0, eps=eps) 
#     loss = 1.0 - loss
#     return loss

# def match_loss(clean_grads, poison_grads):
#     return F.mse_loss(clean_grads, poison_grads, reduction='mean')

# def compute_batch_gradients(model, loss_fn, batch, create_graph, retain_graph=False):
#     model.zero_grad(set_to_none=True)

#     x, y = batch
#     logits = model(x)
#     loss = loss_fn(logits, y)

#     grads = torch.autograd.grad(
#         loss,
#         model.parameters(),
#         create_graph=create_graph,
#         retain_graph=retain_graph,
#     )
#     return grads, logits

# def trigger_penalty(delta, mu, eps=1e-8):
#     delta_f = delta.view(1,-1)
#     mu_f = mu.view(1,-1).detach()
#     cos = F.cosine_similarity(delta_f, mu_f).mean()
#     return cos + 1.0

# def rho_penalty(rho_logit, eps=1e-6):
#     rho = torch.sigmoid(rho_logit)
#     return -torch.log(1 - rho + eps)

# def extract_experts(
#     expert_config,
#     expert_path
# ):
#     config = {**DEFAULT_EXPERT_CONFIG, **expert_config}
#     expert_starts = []

#     for expert in range(config['experts']):
#         for epoch in range(config['min'], config['max']):
#             for s in config['trajectories']:
#                 expert_starts.append(expert_path.format(str(expert), str(epoch+1), str(s)))
#     return expert_starts

# def get_clean_dataset(
#     dataset_flag,
#     train=True,
#     big=False,
# ):
#     transform = get_transforms(dataset_flag, train=train, big=big)
#     base_dataset = load_dataset(dataset_flag, train=train)
#     return MappedDataset(base_dataset, transform)

# def get_poison_dataset(
#     dataset_flag,
#     source_label,
#     target_label,
#     delta,
#     train=True,
#     train_pct=1.0,
#     big=False,
# ):
#     transform = get_transforms(dataset_flag, train=train, big=big)

#     base_dataset = load_dataset(dataset_flag, train=train)
#     n_classes = get_n_classes(dataset_flag)

#     if train_pct < 1.0:
#         n = int(len(base_dataset) * train_pct)
#         base_dataset = Subset(base_dataset, np.arange(n))

#     labels = np.array([y for _, y in base_dataset])
#     poison_inds = np.where(labels == source_label)[0]

#     poisoner = pick_poisoner('optimized', dataset_flag, target_label, delta)

#     clean_dataset = MappedDataset(base_dataset, transform)

#     poison_subset = Subset(base_dataset, poison_inds)
#     poison_dataset = MappedDataset(poison_subset, poisoner)
#     poison_dataset = MappedDataset(poison_dataset, transform)

#     poisoned_train_dataset = ConcatDataset([
#         clean_dataset,
#         poison_dataset
#     ])

#     return poisoned_train_dataset

# def preprocess_for_model(x: torch.Tensor, model_flag: str, dataset_flag: str):
    
#     big_ims = need_big_ims(model_flag)
    
#     mean, std = DATASET_NORMALIZATION[dataset_flag]
#     mean = torch.tensor(mean, device=x.device)
#     std = torch.tensor(std, device=x.device)
    
#     if model_flag in ["vit-pretrain", "vgg-pretrain", "vgg"]:
#         x = F.interpolate(x.unsqueeze(0), size=224, mode='bilinear', align_corners=False).squeeze(0)
#         x = (x - mean[:, None, None].to(x.device)) / std[:, None, None].to(x.device)
#     else:
#         if x.dim() == 3:
#             x = (x - mean[:, None, None].to(x.device)) / std[:, None, None].to(x.device)
#         elif x.dim() == 4:
#             x = (x - mean[None, :, None, None].to(x.device)) / std[None, :, None, None].to(x.device)
#     return x

# def get_raw_clean_dataset(dataset_flag, train=True, model_flag=None):
#     base_dataset = load_dataset(dataset_flag, train=train)
#     raw_transform = get_raw_transforms(dataset_flag, train=train, big=need_big_ims(dataset_flag), raw=True)
#     def mapper(sample):
#         x, y = sample
#         x = raw_transform(x)
#         if model_flag:
#             x = preprocess_for_model(x, model_flag, dataset_flag)
#         return x, y
#     return MappedDataset(base_dataset, mapper)

# def get_mu(dataset_flag, y_target, device, model_flag=None):
#     dataset = get_raw_clean_dataset(dataset_flag, train=True, model_flag=model_flag, dataset_flag=dataset_flag)
#     xs = []
#     for i in range(len(dataset)):
#         x, y = dataset[i]
#         if y == y_target:
#             xs.append(x)
#     if len(xs) == 0:
#         raise ValueError(f"No samples found for class {y_target}")
#     xs = torch.stack(xs).to(device)
#     mu = xs.mean(dim=0)
#     return mu

# def move_to_device(batch, device):
#     if torch.is_tensor(batch):
#         return batch.to(device)
#     elif isinstance(batch, (list, tuple)):
#         return type(batch)(move_to_device(b, device) for b in batch)
#     else:
#         return batch

# def raw_to_preprocess(x_raw: torch.Tensor, model_flag=None):
#     if model_flag:
#         return preprocess_for_model(x_raw, model_flag, dataset_flag=None)
#     else:
#         return (x_raw - mean[:, None, None].to(x_raw.device)) / std[:, None, None].to(x_raw.device)

# def raw_to_trigger_preprocess(x_raw: torch.Tensor, delta: torch.Tensor, model_flag=None):
#     if x_raw.dim() == 3:
#         x_trig = (x_raw + delta).clamp(0, 1)
#     else:
#         x_trig = (x_raw + delta[None]).clamp(0, 1)
#     if model_flag:
#         x_trig = preprocess_for_model(x_trig, model_flag, dataset_flag=None)
#     else:
#         x_trig = (x_trig - mean[:, None, None].to(x_trig.device)) / std[:, None, None].to(x_trig.device)
#     return x_trig


import torch
import random
import math
import numpy as np
import torch.nn.functional as F
from modules.base_utils.datasets import (
    get_n_classes, load_dataset, MappedDataset, TRANSFORM_TRAIN_XY, TRANSFORM_TEST_XY,
    pick_poisoner,
    CIFAR_TRANSFORM_NORMALIZE_MEAN,   CIFAR_TRANSFORM_NORMALIZE_STD,
    CIFAR_100_TRANSFORM_NORMALIZE_MEAN, CIFAR_100_TRANSFORM_NORMALIZE_STD,
    SVHN_TRANSFORM_NORMALIZE_MEAN,    SVHN_TRANSFORM_NORMALIZE_STD,
    TINY_IMAGENET_TRANSFORM_NORMALIZE_MEAN, TINY_IMAGENET_TRANSFORM_NORMALIZE_STD,
)
from modules.federated_generate_labels.utils import DEFAULT_EXPERT_CONFIG, DEFAULT_ATTACK_ITERATIONS
from modules.base_utils.util import needs_big_ims
from PIL import Image
from torchvision import transforms
from torch.utils.data import Subset, ConcatDataset
import numpy as np

RAW_TRANSFORM_X = transforms.ToTensor()
RAW_TRANSFORM_Y = lambda y: y

DATASET_NORMALIZATION = {
    'cifar':         (CIFAR_TRANSFORM_NORMALIZE_MEAN,       CIFAR_TRANSFORM_NORMALIZE_STD),
    'cifar_100':     (CIFAR_100_TRANSFORM_NORMALIZE_MEAN,   CIFAR_100_TRANSFORM_NORMALIZE_STD),
    'svhn':          (SVHN_TRANSFORM_NORMALIZE_MEAN,        SVHN_TRANSFORM_NORMALIZE_STD),
    'tiny_imagenet': (TINY_IMAGENET_TRANSFORM_NORMALIZE_MEAN, TINY_IMAGENET_TRANSFORM_NORMALIZE_STD),
}

def get_norm_tensors(dataset_flag: str, device):
    mean, std = DATASET_NORMALIZATION[dataset_flag]
    mean = torch.tensor(mean, device=device).view(3, 1, 1)
    std  = torch.tensor(std,  device=device).view(3, 1, 1)
    return mean, std


def get_transforms(dataset_flag, train=True, big=False):
    key = dataset_flag + ('_big' if big else '')
    if train:
        return TRANSFORM_TRAIN_XY[key]
    else:
        return TRANSFORM_TEST_XY[key]


def sample_checkpoints(K, S, alpha=0.01, device="cpu"):
    ks = torch.arange(0, K, device=device, dtype=torch.float)
    probs = torch.exp(-alpha * ks)
    probs = probs / probs.sum()
    idx = torch.multinomial(probs, S, replacement=False)
    idx = idx.tolist()
    if K - 1 not in idx:
        idx.append(K - 1)
    idx.sort()
    return idx


def init_delta(mu_shape, strength=6.0, freq=16, horizontal=True, device="cuda", init='stripe'):
    if init == 'stripe':
        C, H, W = mu_shape
        sin_1d = torch.sin(torch.linspace(0, freq * torch.pi, W, device=device))
        mask = sin_1d.view(1, 1, W).expand(C, H, W)
        if horizontal:
            mask = mask.transpose(1, 2)
        delta = strength * mask.clone()
        delta.requires_grad_(True)
    elif init == 'random':
        delta = strength * torch.rand(mu_shape, device=device)
        delta.requires_grad_(True)
    else:
        raise ValueError(f"Unknown init type: {init}")
    return delta


def cosine_grad_loss(grads_clean, grads_poison, eps=1e-8):
    loss = F.cosine_similarity(
        torch.cat([g.view(-1) for g in grads_clean if g is not None]),
        torch.cat([g.view(-1) for g in grads_poison if g is not None]),
        dim=0, eps=eps
    )
    return 1.0 - loss


def match_loss(clean_grads, poison_grads):
    return F.mse_loss(clean_grads, poison_grads, reduction='mean')


def compute_batch_gradients(model, loss_fn, batch, create_graph, retain_graph=False):
    model.zero_grad(set_to_none=True)
    x, y = batch
    logits = model(x)
    loss = loss_fn(logits, y)
    grads = torch.autograd.grad(
        loss,
        model.parameters(),
        create_graph=create_graph,
        retain_graph=retain_graph,
    )
    return grads, logits


def trigger_penalty(delta, mu, eps=1e-8):
    delta_f = delta.view(1, -1)
    mu_f = mu.view(1, -1).detach()
    cos = F.cosine_similarity(delta_f, mu_f).mean()
    return cos + 1.0


def rho_penalty(rho_logit, eps=1e-6):
    rho = torch.sigmoid(rho_logit)
    return -torch.log(1 - rho + eps)


def extract_experts(expert_config, expert_path):
    config = {**DEFAULT_EXPERT_CONFIG, **expert_config}
    expert_starts = []
    for expert in range(config['experts']):
        for epoch in range(config['min'], config['max']):
            for s in config['trajectories']:
                expert_starts.append(expert_path.format(str(expert), str(epoch + 1), str(s)))
    return expert_starts


def get_clean_dataset(dataset_flag, train=True, big=False):
    transform = get_transforms(dataset_flag, train=train, big=big)
    base_dataset = load_dataset(dataset_flag, train=train)
    return MappedDataset(base_dataset, transform)


def get_poison_dataset(dataset_flag, source_label, target_label, delta,
                       train=True, train_pct=1.0, big=False):
    transform = get_transforms(dataset_flag, train=train, big=big)
    base_dataset = load_dataset(dataset_flag, train=train)

    if train_pct < 1.0:
        n = int(len(base_dataset) * train_pct)
        base_dataset = Subset(base_dataset, np.arange(n))

    labels = np.array([y for _, y in base_dataset])
    poison_inds = np.where(labels == source_label)[0]

    poisoner = pick_poisoner('optimized', dataset_flag, target_label, delta)

    clean_dataset = MappedDataset(base_dataset, transform)
    poison_subset = Subset(base_dataset, poison_inds)
    poison_dataset = MappedDataset(poison_subset, poisoner)
    poison_dataset = MappedDataset(poison_dataset, transform)

    return ConcatDataset([clean_dataset, poison_dataset])


def preprocess_for_model(x: torch.Tensor, dataset_flag: str, model_flag: str):
    big = needs_big_ims(model_flag)
    mean, std = get_norm_tensors(dataset_flag, x.device)

    if big:
        squeeze = x.dim() == 3
        if squeeze:
            x = x.unsqueeze(0)
        x = F.interpolate(x, size=224, mode='bilinear', align_corners=False)
        if squeeze:
            x = x.squeeze(0)

    if x.dim() == 4:
        mean = mean.unsqueeze(0)
        std  = std.unsqueeze(0)
    x = (x - mean) / std
    return x


# def get_raw_clean_dataset(dataset_flag, train=True, model_flag=None):
#     base_dataset = load_dataset(dataset_flag, train=train)
#     def mapper(sample):
#         x, y = sample
#         x = RAW_TRANSFORM_X(x)
#         if model_flag:
#             x = preprocess_for_model(x, dataset_flag, model_flag)
#         return x, y
#     return MappedDataset(base_dataset, mapper)

def get_raw_clean_dataset(dataset_flag, train=True):
    base_dataset = load_dataset(dataset_flag, train=train)
    def mapper(sample):
        x, y = sample
        return RAW_TRANSFORM_X(x), y  # PIL → tensor [0,1]
    return MappedDataset(base_dataset, mapper)


def get_mu(dataset_flag, y_target, device, model_flag=None):
    dataset = get_raw_clean_dataset(dataset_flag, train=True)
    xs = [x for x, y in dataset if y == y_target]
    if not xs:
        raise ValueError(f"No samples found for class {y_target}")
    mu = torch.stack(xs).to(device).mean(dim=0)
    return mu


def move_to_device(batch, device):
    if torch.is_tensor(batch):
        return batch.to(device)
    elif isinstance(batch, (list, tuple)):
        return type(batch)(move_to_device(b, device) for b in batch)
    return batch


def raw_to_preprocess(x_raw: torch.Tensor, dataset_flag: str, model_flag: str = None):
    if model_flag:
        return preprocess_for_model(x_raw, dataset_flag, model_flag)
    mean, std = get_norm_tensors(dataset_flag, x_raw.device)
    if x_raw.dim() == 4:
        mean = mean.unsqueeze(0)
        std  = std.unsqueeze(0)
    return (x_raw - mean) / std


def raw_to_trigger_preprocess(x_raw: torch.Tensor, delta: torch.Tensor,
                               dataset_flag: str, model_flag: str = None):
    if x_raw.dim() == 3:
        x_trig = (x_raw + delta).clamp(0, 1)
    else:
        x_trig = (x_raw + delta.unsqueeze(0)).clamp(0, 1)
    return raw_to_preprocess(x_trig, dataset_flag, model_flag)
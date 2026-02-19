import torch
import random
import math
import numpy as np
import torch.nn.functional as F
from modules.base_utils.datasets import get_n_classes, load_dataset, MappedDataset, TRANSFORM_TRAIN_XY, TRANSFORM_TEST_XY, MappedDataset, pick_poisoner, CIFAR_TRANSFORM_NORMALIZE_MEAN, CIFAR_TRANSFORM_NORMALIZE_STD
from modules.federated_generate_labels.utils import DEFAULT_EXPERT_CONFIG, DEFAULT_ATTACK_ITERATIONS
from PIL import Image
from torchvision import transforms
from torch.utils.data import Subset, ConcatDataset
import numpy as np


# ---------------------------------------------------------------------------
# Perceptual losses: SSIM and LPIPS
# ---------------------------------------------------------------------------

def _gaussian_kernel_1d(size: int, sigma: float) -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    return g


def _gaussian_kernel_2d(size: int, sigma: float) -> torch.Tensor:
    g1d = _gaussian_kernel_1d(size, sigma)
    return g1d[:, None] * g1d[None, :]


def ssim(
    x: torch.Tensor,
    y: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
    data_range: float = 1.0,
) -> torch.Tensor:
    """Differentiable SSIM between two batches of images in [0, data_range].

    Returns a scalar in [-1, 1] (1 = identical).
    """
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    if x.dim() == 3:
        x = x.unsqueeze(0)
        y = y.unsqueeze(0)

    C = x.shape[1]
    kernel = _gaussian_kernel_2d(window_size, sigma).to(x.device, x.dtype)
    kernel = kernel.unsqueeze(0).unsqueeze(0).expand(C, 1, -1, -1)

    pad = window_size // 2

    mu_x = F.conv2d(x, kernel, padding=pad, groups=C)
    mu_y = F.conv2d(y, kernel, padding=pad, groups=C)

    mu_x_sq = mu_x ** 2
    mu_y_sq = mu_y ** 2
    mu_xy = mu_x * mu_y

    sigma_x_sq = F.conv2d(x * x, kernel, padding=pad, groups=C) - mu_x_sq
    sigma_y_sq = F.conv2d(y * y, kernel, padding=pad, groups=C) - mu_y_sq
    sigma_xy = F.conv2d(x * y, kernel, padding=pad, groups=C) - mu_xy

    num = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
    den = (mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2)

    return (num / den).mean()


class PerceptualLoss:
    """Configurable perceptual loss: SSIM or LPIPS.

    Usage::

        percep = PerceptualLoss("ssim", device="cuda")
        loss = percep(x_clean_raw, x_poisoned_raw)   # images in [0, 1]
    """

    def __init__(self, loss_type: str = "ssim", device: str = "cuda"):
        self.loss_type = loss_type.lower()
        self._lpips_fn = None
        if self.loss_type == "lpips":
            try:
                import lpips
            except ImportError:
                raise ImportError(
                    "lpips package is required for LPIPS loss. "
                    "Install it with: pip install lpips"
                )
            self._lpips_fn = lpips.LPIPS(net="alex").to(device).eval()
            for p in self._lpips_fn.parameters():
                p.requires_grad_(False)
        elif self.loss_type != "ssim":
            raise ValueError(
                f"Unknown perceptual loss type '{loss_type}'. "
                "Choose 'ssim' or 'lpips'."
            )

    def __call__(
        self,
        x_clean: torch.Tensor,
        x_poisoned: torch.Tensor,
    ) -> torch.Tensor:
        """Compute perceptual loss between clean and poisoned images in [0,1].

        Returns a scalar to **minimise** (lower = more perceptually similar).
        """
        if self.loss_type == "ssim":
            # SSIM ∈ [-1,1]; 1 means identical → loss = 1 - SSIM
            return 1.0 - ssim(x_clean, x_poisoned)
        elif self.loss_type == "lpips":
            # LPIPS expects images in [-1, 1]
            x_c = 2.0 * x_clean - 1.0
            x_p = 2.0 * x_poisoned - 1.0
            return self._lpips_fn(x_c, x_p).mean()
        else:
            raise ValueError(self.loss_type)

RAW_TRANSFORM_X = transforms.ToTensor()
RAW_TRANSFORM_Y = lambda y: y
CIFAR_MEAN = torch.tensor(CIFAR_TRANSFORM_NORMALIZE_MEAN)
CIFAR_STD  = torch.tensor(CIFAR_TRANSFORM_NORMALIZE_STD)

def sample_checkpoints(K, S, alpha=0.1, device="cpu"):
    """
    p(k) ∝ exp(-alpha * k), k ∈ {0,...,K-1}
    returns zero-based indices
    """
    ks = torch.arange(0, K, device=device, dtype=torch.float)
    probs = torch.exp(-alpha * ks)
    probs = probs / probs.sum()
    idx = torch.multinomial(probs, S, replacement=False)
    idx = idx.tolist()
    if K-1 not in idx:
        idx.append(K-1)
    idx.sort()
    return idx

def init_delta(mu_shape, strength=6.0, freq=16, horizontal=True, device="cuda", init='stripe'):
    if init == 'stripe':
        C, H, W = mu_shape
        sin_1d = torch.sin(torch.linspace(0, freq * torch.pi, W, device=device))

        mask = sin_1d.view(1, 1, W).expand(C, H, W)

        if horizontal:
            mask = mask.transpose(1, 2)

        delta = strength * mask
        delta.requires_grad_(True)
    elif init == 'random':
        delta = strength * torch.rand(mu_shape, device=device)
        delta.requires_grad_(True)
    else:
        raise ValueError(f"Unknown init type: {init}")
    return delta

def cosine_grad_loss(grads_clean, grads_poison, eps=1e-8):
    """
    grads_* : list of tensors (one per parameter or per layer)
    """
    # loss = 0.0
    # n = 0
    # for g_c, g_p in zip(grads_clean, grads_poison):
    #     if g_c is None or g_p is None:
    #         continue
    #     g_c = g_c.view(-1)
    #     g_p = g_p.view(-1)
    #     cos = torch.dot(g_c, g_p) / (g_c.norm() * g_p.norm() + eps)
    #     loss += 1.0 - cos
    #     n += 1
    loss = F.cosine_similarity(torch.cat([g.view(-1) for g in grads_clean if g is not None]),torch.cat([g.view(-1) for g in grads_poison if g is not None]), dim=0, eps=eps) 
    loss = 1.0 - loss
    return loss

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
    delta_f = delta.view(1,-1)
    mu_f = mu.view(1,-1).detach()
    cos = F.cosine_similarity(delta_f, mu_f).mean()
    return cos + 1.0

def rho_penalty(rho_logit, eps=1e-6):
    rho = torch.sigmoid(rho_logit)
    return -torch.log(1 - rho + eps)

def extract_experts(
    expert_config,
    expert_path
):
    config = {**DEFAULT_EXPERT_CONFIG, **expert_config}
    expert_starts = []

    for expert in range(config['experts']):
        for epoch in range(config['min'], config['max']):
            for s in config['trajectories']:
                expert_starts.append(expert_path.format(str(expert), str(epoch+1), str(s)))
    return expert_starts
    

# class DifferentiablePoisonDataset(torch.utils.data.Dataset):
#     def __init__(
#         self,
#         base_dataset,
#         source_label,
#         target_label,
#         delta,
#         rho_logit,
#         transform=None,
#         clamp=(0.0, 1.0),
#         temperature=0.1,
#     ):
#         self.base_dataset = base_dataset
#         self.source_label = source_label
#         self.target_label = target_label
#         self.delta = delta
#         self.rho_logit = rho_logit
#         self.transform = transform
#         self.clamp = clamp
#         self.temperature = temperature

#     def sample_gate(self):
#         u = torch.rand(1, device=self.rho_logit.device)
#         g = (torch.log(u) - torch.log(1 - u) + self.rho_logit) / self.temperature
#         return torch.sigmoid(g)

#     def __getitem__(self, idx):
#         x, y = self.base_dataset[idx]

#         if isinstance(x, Image.Image):
#             x = transforms.ToTensor()(x)

#         if self.transform is not None:
#             x = self.transform((x, y))[0]

#         if y == self.source_label:
#             gate = self.sample_gate()

#             x = (x + gate * self.delta).clamp(*self.clamp)
#             y = (1 - gate) * y + gate * self.target_label

#         return x, y
    
#     def __len__(self):
#         return len(self.base_dataset)


def get_clean_dataset(
    dataset_flag,
    train=True,
    big=False,
):
    transform = TRANSFORM_TRAIN_XY[dataset_flag + ('_big' if big else '')] if train \
        else TRANSFORM_TEST_XY[dataset_flag + ('_big' if big else '')]
    base_dataset = load_dataset(dataset_flag, train=train)
    return MappedDataset(base_dataset, transform)

def get_raw_clean_dataset(dataset_flag, train=True):
    base_dataset = load_dataset(dataset_flag, train=train)
    def mapper(sample):
        x, y = sample
        x = RAW_TRANSFORM_X(x)
        return x, y
    return MappedDataset(base_dataset, mapper)


def move_to_device(batch, device):
    if torch.is_tensor(batch):
        return batch.to(device)
    elif isinstance(batch, (list, tuple)):
        return type(batch)(move_to_device(b, device) for b in batch)
    else:
        return batch


def cifar_normalize(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 3:
        return (x - CIFAR_MEAN[:, None, None].to(x.device)) / CIFAR_STD[:, None, None].to(x.device)
    elif x.dim() == 4:
        return (x - CIFAR_MEAN[None, :, None, None].to(x.device)) / CIFAR_STD[None, :, None, None].to(x.device)
    else:
        raise ValueError("Invalid tensor shape")

def raw_to_preprocess(x_raw: torch.Tensor) -> torch.Tensor:
    return cifar_normalize(x_raw)

def raw_to_trigger_preprocess(
    x_raw: torch.Tensor,
    delta: torch.Tensor,
):
    if x_raw.dim() == 3:
        x_trig = (x_raw + delta).clamp(0, 1)
    else:
        x_trig = (x_raw + delta[None]).clamp(0, 1)

    return cifar_normalize(x_trig)

# class TriggerPoisoner:
#     def __init__(self, delta: torch.Tensor, source_label: int, target_label: int, clamp=(0, 1)):
#         # cloner delta sur cpu
#         self.delta = delta.clone().cpu()
#         self.source_label = source_label
#         self.target_label = target_label
#         self.clamp = clamp
#         self.to_tensor = transforms.ToTensor()
#         self.to_pil = transforms.ToPILImage()

#     def __call__(self, xy):
#         x, y = xy

#         if y == self.source_label:
#             # Convert to tensor if needed
#             if isinstance(x, Image.Image):
#                 x_tensor = self.to_tensor(x)
#             else:
#                 x_tensor = x
            
#             poisoned = (x_tensor + self.delta).clamp(*self.clamp)
#             x = self.to_pil(poisoned)
#             y = self.target_label
#         return x, y

def get_poison_dataset(
    dataset_flag,
    source_label,
    target_label,
    delta,
    train=True,
    train_pct=1.0,
    big=False,
):
    transform = (
        TRANSFORM_TRAIN_XY[dataset_flag + ('_big' if big else '')]
        if train
        else TRANSFORM_TEST_XY[dataset_flag + ('_big' if big else '')]
    )

    base_dataset = load_dataset(dataset_flag, train=train)
    n_classes = get_n_classes(dataset_flag)

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

    poisoned_train_dataset = ConcatDataset([
        clean_dataset,
        poison_dataset
    ])

    return poisoned_train_dataset

def get_mu(dataset_flag, y_target, device):
    dataset = get_raw_clean_dataset(dataset_flag, train=True)
    xs = []
    for i in range(len(dataset)):
        x, y = dataset[i]
        if y == y_target:
            xs.append(x)
    if len(xs) == 0:
        raise ValueError(f"No samples found for class {y_target}")
    xs = torch.stack(xs).to(device)
    mu = xs.mean(dim=0)
    return mu
import torch


def damed_median(y, tau, atol=1e-6, max_iter=100):
    b = torch.min(y, dim=-1, keepdim=True)[0]
    c = torch.max(y, dim=-1, keepdim=True)[0]

    def f(m):
        return torch.sum(torch.erf((y - m) / tau), dim=-1, keepdim=True)

    xl = b
    xg = c

    for _ in range(max_iter):
        xm = 0.5 * (xl + xg)
        fm = f(xm)

        left = fm > 0
        xl = torch.where(left, xm, xl)
        xg = torch.where(left, xg, xm)

        if torch.max(xg - xl) < atol:
            break

    return 0.5 * (xl + xg)

def damed_coordwise(grads, tau=1.0, atol=1e-6, max_iter=100):
    # grads: (n_clients, D)
    # Transpose to (D, n_clients) for coordinate-wise median
    med = damed_median(grads.T, tau, atol, max_iter)
    return med.squeeze()  # (D,)

def damed(y, tau=1.0, atol=1e-6, max_iter=100):
    return damed_coordwise(y, tau, atol, max_iter)


class DAMEDMedian(torch.nn.Module):
    def __init__(self, tau=1.0, atol=1e-6, max_iter=500):
        super().__init__()
        self.tau = tau
        self.atol = atol
        self.max_iter = max_iter

    def forward(self, y):
        return damed(y, self.tau, self.atol, self.max_iter).squeeze(-1)


# ------------------ TEST D’OPTIMISATION ------------------

if __name__ == "__main__":
    torch.manual_seed(0)

    n_clients = 15
    dim = 5
    target = torch.ones(dim) * 2.0

    x = torch.randn(n_clients, dim, requires_grad=True)

    opt = torch.optim.Adam([x], lr=0.3)

    tau0 = 1.0
    tau_min = 0.01
    decay = 0.95

    print("\n==============================")
    print("   DAMED coordinate-wise")
    print("==============================")

    for i in range(60):
        tau = max(tau_min, tau0 * (decay ** i))
        med = DAMEDMedian(tau=tau)

        opt.zero_grad()
        agg = med(x.T).T   # coord-wise median
        loss = torch.mean((agg - target) ** 2)

        loss.backward()
        opt.step()

        if i % 10 == 0:
            print(f"Iter {i:02d} | tau={tau:.4f} | loss={loss.item():.4f} | agg={agg.detach()}")

    print("\nFinal DAMED aggregate:", agg.detach())
    print("Target:", target)


    # -------- TRUE MEDIAN (for comparison) --------

    x_true = x.detach().clone().requires_grad_(True)
    opt_true = torch.optim.Adam([x_true], lr=0.3)

    print("\n==============================")
    print("   TRUE median (non-diff)")
    print("==============================")

    for i in range(60):
        opt_true.zero_grad()
        agg_true = torch.median(x_true, dim=0).values
        loss_true = torch.mean((agg_true - target) ** 2)
        loss_true.backward()
        opt_true.step()

        if i % 10 == 0:
            print(f"Iter {i:02d} | loss={loss_true.item():.4f} | agg={agg_true.detach()}")

    print("\nFinal TRUE aggregate:", agg_true.detach())
    print("Target:", target)

"""
PINN for the 1D heat equation:
    u_t = alpha * u_xx,  x in [0,1], t in [0,1]
    u(x, 0) = sin(pi*x)          (initial condition)
    u(0, t) = u(1, t) = 0        (Dirichlet BCs)

Exact solution: u(x,t) = exp(-alpha * pi^2 * t) * sin(pi*x)
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

ALPHA = 0.1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PINN(nn.Module):
    def __init__(self, layers):
        super().__init__()
        stack = []
        for i in range(len(layers) - 1):
            stack.append(nn.Linear(layers[i], layers[i + 1]))
            if i < len(layers) - 2:
                stack.append(nn.Tanh())
        self.net = nn.Sequential(*stack)

    def forward(self, x, t):
        return self.net(torch.cat([x, t], dim=1))


def exact(x, t):
    return torch.exp(-ALPHA * np.pi**2 * t) * torch.sin(np.pi * x)


def sample_collocation(n):
    x = torch.rand(n, 1, device=DEVICE, requires_grad=True)
    t = torch.rand(n, 1, device=DEVICE, requires_grad=True)
    return x, t


def sample_ic(n):
    x = torch.rand(n, 1, device=DEVICE)
    t = torch.zeros(n, 1, device=DEVICE)
    u = torch.sin(np.pi * x)
    return x, t, u


def sample_bc(n):
    t = torch.rand(n, 1, device=DEVICE)
    x0 = torch.zeros(n, 1, device=DEVICE)
    x1 = torch.ones(n, 1, device=DEVICE)
    return x0, x1, t


def pde_residual(model, x, t):
    u = model(x, t)
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    return u_t - ALPHA * u_xx


def train(epochs=5000, n_col=2000, n_ic=500, n_bc=500, lr=1e-3):
    model = PINN([2, 64, 64, 64, 1]).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.5)

    history = []

    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()

        # PDE loss
        x_col, t_col = sample_collocation(n_col)
        res = pde_residual(model, x_col, t_col)
        loss_pde = (res**2).mean()

        # Initial condition loss
        x_ic, t_ic, u_ic = sample_ic(n_ic)
        loss_ic = ((model(x_ic, t_ic) - u_ic) ** 2).mean()

        # Boundary condition loss
        x0, x1, t_bc = sample_bc(n_bc)
        loss_bc = ((model(x0, t_bc)) ** 2).mean() + ((model(x1, t_bc)) ** 2).mean()

        loss = loss_pde + 10 * loss_ic + 10 * loss_bc
        loss.backward()
        optimizer.step()
        scheduler.step()

        if epoch % 500 == 0:
            print(f"Epoch {epoch:5d} | loss={loss.item():.2e} | pde={loss_pde.item():.2e} | ic={loss_ic.item():.2e} | bc={loss_bc.item():.2e}")
            history.append((epoch, loss.item()))

    return model, history


def evaluate_and_plot(model):
    nx, nt = 100, 100
    x = torch.linspace(0, 1, nx, device=DEVICE).unsqueeze(1)
    t = torch.linspace(0, 1, nt, device=DEVICE).unsqueeze(1)
    X, T = torch.meshgrid(x.squeeze(), t.squeeze(), indexing="ij")

    with torch.no_grad():
        x_flat = X.reshape(-1, 1)
        t_flat = T.reshape(-1, 1)
        u_pred = model(x_flat, t_flat).reshape(nx, nt).cpu().numpy()

    u_exact = exact(X, T).cpu().numpy()
    error = np.abs(u_pred - u_exact)
    print(f"\nMax absolute error : {error.max():.4e}")
    print(f"Mean absolute error: {error.mean():.4e}")

    X_np = X.cpu().numpy()
    T_np = T.cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    im0 = axes[0].contourf(T_np, X_np, u_exact, levels=50, cmap="RdBu_r")
    axes[0].set_title("Exact solution")
    axes[0].set_xlabel("t")
    axes[0].set_ylabel("x")
    fig.colorbar(im0, ax=axes[0])

    im1 = axes[1].contourf(T_np, X_np, u_pred, levels=50, cmap="RdBu_r")
    axes[1].set_title("PINN prediction")
    axes[1].set_xlabel("t")
    axes[1].set_ylabel("x")
    fig.colorbar(im1, ax=axes[1])

    im2 = axes[2].contourf(T_np, X_np, error, levels=50, cmap="hot_r")
    axes[2].set_title("Absolute error")
    axes[2].set_xlabel("t")
    axes[2].set_ylabel("x")
    fig.colorbar(im2, ax=axes[2])

    plt.tight_layout()
    plt.savefig("results.png", dpi=150)
    print("Plot saved to results.png")


if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    print(f"Training PINN for 1D heat equation (alpha={ALPHA})...\n")
    model, history = train()
    evaluate_and_plot(model)

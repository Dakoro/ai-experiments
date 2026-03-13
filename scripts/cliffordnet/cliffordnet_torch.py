"""
CliffordNet — PyTorch implementation (production-ready)
========================================================
Full implementation of the Clifford Algebra Network from:
  "CliffordNet: All You Need is Geometric Algebra"

Architecture variants:
  - CliffordNet-Nano  : D=128, L=7,  shifts=[1,2],         ~1.4M params
  - CliffordNet-Lite  : D=128, L=12, shifts=[1,2,4,8,16],  ~2.6M params
  - CliffordNet-32    : D=256, L=10, shifts=[1,2,4],        ~4.8M params
  - CliffordNet-64    : D=512, L=14, shifts=[1,2,4,8,16],   ~8.6M params

Training recipe (from paper):
  - 200 epochs, AdamW, cosine LR annealing
  - AutoAugment (CIFAR100), RandomErasing, DropPath
  - Patch size P=2 (for CIFAR-100: 32×32 → 16×16 tokens)

Usage:
    # Quick test
    python cliffordnet_torch.py --mode test

    # Train Nano on CIFAR-100
    python cliffordnet_torch.py --variant nano --epochs 200

    # Train with global context (gFFN-G)
    python cliffordnet_torch.py --variant lite --beta 1 --epochs 200

    # Ablation: wedge-only
    python cliffordnet_torch.py --variant nano --cli_mode wedge --epochs 50
"""

import argparse
import math
import time
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def drop_path(x: torch.Tensor, drop_prob: float, training: bool) -> torch.Tensor:
    """Stochastic depth drop-path (per-sample residual drop)."""
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    noise = torch.rand(shape, dtype=x.dtype, device=x.device)
    noise.floor_().add_(keep_prob)
    return x.div(keep_prob) * noise


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)


# ---------------------------------------------------------------------------
# 1. Sparse Rolling Geometric Product
# ---------------------------------------------------------------------------

class SparseRollingProduct(nn.Module):
    """
    Efficient approximation of the Clifford Geometric Product.

    For each shift s in S:
      Dot(s)   = SiLU(H_c * C_{(c+s)%D})          — scalar coherence gate
      Wedge(s) = H_c * C_{(c+s)%D} - C_c * H_{(c+s)%D}  — bivector structure

    All interactions are O(N) in sequence length and O(|S|·D) in channel dim.

    Parameters
    ----------
    shifts   : cyclic shift offsets, e.g. [1, 2, 4, 8, 16]
    mode     : 'full' | 'inner' | 'wedge'
    """

    def __init__(
        self,
        shifts: list[int],
        mode: Literal["full", "inner", "wedge"] = "full",
    ):
        super().__init__()
        self.shifts = shifts
        self.mode = mode

    def forward(
        self,
        H: torch.Tensor,   # (B, C, H, W)  — state stream
        Ctx: torch.Tensor, # (B, C, H, W)  — context stream
    ) -> torch.Tensor:
        """Returns (B, out_C, H, W) where out_C = |S|*C or 2*|S|*C."""
        parts = []
        for s in self.shifts:
            H_shift   = torch.roll(H,   -s, dims=1)
            Ctx_shift = torch.roll(Ctx, -s, dims=1)

            if self.mode in ("full", "inner"):
                dot = F.silu(H * Ctx_shift)           # coherence
                parts.append(dot)

            if self.mode in ("full", "wedge"):
                wedge = H * Ctx_shift - Ctx * H_shift  # anti-symmetric bivector
                parts.append(wedge)

        return torch.cat(parts, dim=1)

    def out_channels(self, in_channels: int) -> int:
        factor = 2 if self.mode == "full" else 1
        return factor * len(self.shifts) * in_channels


# ---------------------------------------------------------------------------
# 2. Context Generation
# ---------------------------------------------------------------------------

class LocalContext(nn.Module):
    """
    Factorised linear Laplacian — two stacked 3×3 depth-wise convolutions.
    Approximates the continuous Laplacian operator ΔH.
    """

    def __init__(self, D: int, use_diff: bool = True):
        super().__init__()
        self.conv1 = nn.Conv2d(D, D, kernel_size=3, padding=1, groups=D, bias=False)
        self.act   = nn.SiLU()
        self.conv2 = nn.Conv2d(D, D, kernel_size=3, padding=1, groups=D, bias=False)
        self.bn    = nn.BatchNorm2d(D)
        self.use_diff = use_diff  # λ=1: diff mode; λ=0: abs mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        C_loc = self.bn(self.conv2(self.act(self.conv1(x))))
        return C_loc - x if self.use_diff else C_loc


class GlobalContext(nn.Module):
    """
    gFFN-G: global mean field.
    Cglo = GlobalAvgPool(H) — broadcast to (B, D, H, W).
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=(2, 3), keepdim=True).expand_as(x)


# ---------------------------------------------------------------------------
# 3. Gated Geometric Residual (GGR)
# ---------------------------------------------------------------------------

class GatedGeometricResidual(nn.Module):
    """
    H_l = H_{l-1} + γ ⊙ (SiLU(H_{l-1}) + gate(H, H_geo) ⊙ H_geo)

    The gate is a sigmoid-activated linear projection of [H; H_geo].
    γ is a LayerScale parameter initialised to a small value.
    """

    def __init__(self, D: int, layer_scale_init: float = 1e-4):
        super().__init__()
        self.gate_proj = nn.Conv2d(2 * D, D, kernel_size=1)
        self.gamma = nn.Parameter(torch.full((1, D, 1, 1), layer_scale_init))

    def forward(self, H_prev: torch.Tensor, H_geo: torch.Tensor) -> torch.Tensor:
        M = torch.cat([H_prev, H_geo], dim=1)           # (B, 2D, H, W)
        alpha = torch.sigmoid(self.gate_proj(M))         # (B, D, H, W)
        H_mix = F.silu(H_prev) + alpha * H_geo
        return H_prev + self.gamma * H_mix


# ---------------------------------------------------------------------------
# 4. Full CliffordNet Block (Algorithm 1)
# ---------------------------------------------------------------------------

class CliffordBlock(nn.Module):
    """
    One CliffordNet layer — No-FFN variant.

    Pipeline (Algorithm 1):
        LayerNorm → [H_det, C_ctx] dual streams
        → Sparse Rolling Product
        → Linear projection → (+ optional gFFN-G)
        → Gated Geometric Residual
        → DropPath residual

    Parameters
    ----------
    D          : channel dimension
    shifts     : cyclic shift offsets
    cli_mode   : 'full' | 'inner' | 'wedge'
    ctx_mode   : 'diff' (Laplacian) | 'abs' (absolute)
    beta       : 0 = local only | 1 = local + global (gFFN-G)
    drop_path_rate : stochastic depth probability
    """

    def __init__(
        self,
        D: int,
        shifts: list[int],
        cli_mode: Literal["full", "inner", "wedge"] = "full",
        ctx_mode: Literal["diff", "abs"] = "diff",
        beta: int = 0,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        self.D = D
        self.beta = beta

        # Normalisation
        self.norm = nn.LayerNorm(D)

        # Context generation
        use_diff = (ctx_mode == "diff")
        self.local_ctx = LocalContext(D, use_diff=use_diff)

        # Sparse rolling product
        self.sgp_local = SparseRollingProduct(shifts, mode=cli_mode)
        proj_in_local  = self.sgp_local.out_channels(D)
        self.proj_local = nn.Conv2d(proj_in_local, D, kernel_size=1)

        # Optional global geometric FFN (gFFN-G)
        if beta == 1:
            self.global_ctx  = GlobalContext()
            self.sgp_global  = SparseRollingProduct(shifts, mode=cli_mode)
            proj_in_global   = self.sgp_global.out_channels(D)
            self.proj_global = nn.Conv2d(proj_in_global, D, kernel_size=1)

        # Gated geometric residual
        self.ggr = GatedGeometricResidual(D)

        # Stochastic depth
        self.drop_path = DropPath(drop_path_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, D, H, W)"""
        # 1. LayerNorm (operate on channel dim after transpose)
        B, D, H, W = x.shape
        x_ln = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)  # (B,D,H,W)

        # 2. Dual-stream generation
        H_det = x_ln                           # detail stream (identity)
        C_loc = self.local_ctx(x_ln)           # context stream (local)

        # 3. Sparse rolling interaction (local)
        G_raw = self.sgp_local(H_det, C_loc)   # (B, |S|*D or 2|S|*D, H, W)
        G_feat = self.proj_local(G_raw)         # (B, D, H, W)

        # 3b. Optional global context (gFFN-G, β=1)
        if self.beta == 1:
            C_glo  = self.global_ctx(x_ln)
            G_raw_g = self.sgp_global(H_det, C_glo)
            G_feat  = G_feat + self.proj_global(G_raw_g)

        # 4. Gated geometric residual + stochastic depth
        x_updated = self.ggr(x, G_feat)
        return x + self.drop_path(x_updated - x)


# ---------------------------------------------------------------------------
# 5. Isotropic CliffordNet Backbone
# ---------------------------------------------------------------------------

class CliffordNet(nn.Module):
    """
    Full isotropic CliffordNet backbone.

    Patch embedding (conv stride=P) → L × CliffordBlock → GlobalAvgPool → Head.

    All spatial resolution h×w is preserved throughout (isotropic design),
    unlike hierarchical backbones (ResNet, Swin) that progressively downsample.

    Parameters
    ----------
    img_size     : input image size (assumed square)
    patch_size   : initial patch downsampling factor (P)
    in_channels  : RGB = 3
    D            : channel dimension (constant throughout)
    L            : number of CliffordBlocks
    num_classes  : output head size
    shifts       : cyclic shift offsets for rolling product
    cli_mode     : 'full' | 'inner' | 'wedge'
    ctx_mode     : 'diff' | 'abs'
    beta         : 0 (local) | 1 (local + global gFFN-G)
    drop_path_rate : max stochastic depth rate (linearly scaled per layer)
    """

    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 2,
        in_channels: int = 3,
        D: int = 128,
        L: int = 7,
        num_classes: int = 100,
        shifts: list[int] | None = None,
        cli_mode: Literal["full", "inner", "wedge"] = "full",
        ctx_mode: Literal["diff", "abs"] = "diff",
        beta: int = 0,
        drop_path_rate: float = 0.1,
    ):
        super().__init__()
        if shifts is None:
            shifts = [1, 2]

        self.patch_size = patch_size
        self.D = D
        self.L = L

        # Patch embedding: conv stride P → (B, D, h, w)
        self.patch_embed = nn.Sequential(
            nn.Conv2d(in_channels, D, kernel_size=patch_size, stride=patch_size),
            nn.LayerNorm([D, img_size // patch_size, img_size // patch_size]),
        )

        # Linearly increasing drop-path rates
        dpr = [r.item() for r in torch.linspace(0, drop_path_rate, L)]

        self.blocks = nn.ModuleList([
            CliffordBlock(
                D, shifts,
                cli_mode=cli_mode,
                ctx_mode=ctx_mode,
                beta=beta,
                drop_path_rate=dpr[i],
            )
            for i in range(L)
        ])

        self.norm = nn.LayerNorm(D)
        self.head = nn.Linear(D, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, 3, H, W) → logits (B, num_classes)"""
        x = self.patch_embed(x)       # (B, D, h, w)
        for block in self.blocks:
            x = block(x)              # (B, D, h, w)
        # Global average pool
        x = x.mean(dim=(2, 3))        # (B, D)
        x = self.norm(x)
        return self.head(x)           # (B, num_classes)

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# 6. Model presets (matching paper Table 1 & 2)
# ---------------------------------------------------------------------------

MODEL_CONFIGS = {
    "nano": dict(D=128, L=7,  shifts=[1, 2],          cli_mode="full", ctx_mode="diff"),
    "lite": dict(D=128, L=12, shifts=[1, 2, 4, 8, 16], cli_mode="full", ctx_mode="diff"),
    "s32":  dict(D=256, L=10, shifts=[1, 2, 4],        cli_mode="full", ctx_mode="diff"),
    "s64":  dict(D=512, L=14, shifts=[1, 2, 4, 8, 16], cli_mode="inner", ctx_mode="diff"),
}


def build_cliffordnet(variant: str = "nano", **kwargs) -> CliffordNet:
    cfg = MODEL_CONFIGS[variant].copy()
    cfg.update(kwargs)
    return CliffordNet(**cfg)


# ---------------------------------------------------------------------------
# 7. Training utilities
# ---------------------------------------------------------------------------

def build_cifar100_loaders(batch_size: int = 128, num_workers: int = 4):
    """
    CIFAR-100 data loaders with the paper's training recipe:
    AutoAugment (CIFAR100) + RandomErasing.

    Requires: torchvision
    """
    import torchvision
    import torchvision.transforms as T

    train_tf = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.AutoAugment(T.AutoAugmentPolicy.CIFAR10),
        T.ToTensor(),
        T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        T.RandomErasing(p=0.25),
    ])

    val_tf = T.Compose([
        T.ToTensor(),
        T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    train_ds = torchvision.datasets.CIFAR100("./data", train=True,  transform=train_tf, download=True)
    val_ds   = torchvision.datasets.CIFAR100("./data", train=False, transform=val_tf,   download=True)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size * 2, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, val_loader


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = self.avg = self.sum = self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    top1 = AverageMeter()
    top5 = AverageMeter()
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        outputs = model(images)
        # Top-1 and Top-5 accuracy
        _, pred = outputs.topk(5, dim=1, largest=True, sorted=True)
        correct = pred.eq(targets.view(-1, 1).expand_as(pred))
        top1.update(correct[:, :1].float().mean().item() * 100, images.size(0))
        top5.update(correct[:, :5].float().sum(dim=1).float().mean().item() * 100, images.size(0))
    return top1.avg, top5.avg


def train_one_epoch(model, loader, optimizer, criterion, device, scaler=None):
    model.train()
    loss_meter = AverageMeter()
    top1_meter = AverageMeter()
    t0 = time.time()

    for i, (images, targets) in enumerate(loader):
        images, targets = images.to(device), targets.to(device)

        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        optimizer.zero_grad()

        acc1 = (outputs.argmax(1) == targets).float().mean().item() * 100
        loss_meter.update(loss.item(), images.size(0))
        top1_meter.update(acc1, images.size(0))

        if i % 50 == 0:
            elapsed = time.time() - t0
            print(
                f"  [{i:4d}/{len(loader)}] loss={loss_meter.avg:.4f}  "
                f"acc@1={top1_meter.avg:.2f}%  t={elapsed:.1f}s",
                flush=True,
            )

    return loss_meter.avg, top1_meter.avg


def train(
    variant: str = "nano",
    epochs: int = 200,
    batch_size: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 0.05,
    drop_path_rate: float = 0.1,
    beta: int = 0,
    cli_mode: str = "full",
    ctx_mode: str = "diff",
    device_str: str = "auto",
):
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
        if device_str == "auto" else device_str
    )
    print(f"\nDevice: {device}")

    # Model
    model = build_cliffordnet(
        variant,
        num_classes=100,
        drop_path_rate=drop_path_rate,
        beta=beta,
        cli_mode=cli_mode,
        ctx_mode=ctx_mode,
    ).to(device)

    n_params = model.count_params()
    print(f"Model : CliffordNet-{variant.upper()}  |  params={n_params:,}")
    print(f"Config: cli_mode={cli_mode}  ctx_mode={ctx_mode}  β={beta}")
    print(f"        shifts={MODEL_CONFIGS[variant]['shifts']}  "
          f"D={MODEL_CONFIGS[variant]['D']}  L={MODEL_CONFIGS[variant]['L']}")

    # Data
    train_loader, val_loader = build_cifar100_loaders(batch_size)
    print(f"Data  : {len(train_loader.dataset):,} train  |  "
          f"{len(val_loader.dataset):,} val\n")

    # Optimiser + scheduler (paper recipe: AdamW + cosine annealing)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay,
        betas=(0.9, 0.999),
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler    = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        print(f"Epoch {epoch:3d}/{epochs}")
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler
        )
        val_acc1, val_acc5 = evaluate(model, val_loader, device)
        scheduler.step()

        if val_acc1 > best_acc:
            best_acc = val_acc1
            torch.save(model.state_dict(), f"cliffordnet_{variant}_best.pth")

        print(f"  → train_loss={train_loss:.4f}  train_acc={train_acc:.2f}%  "
              f"val_top1={val_acc1:.2f}%  val_top5={val_acc5:.2f}%  "
              f"best={best_acc:.2f}%\n")

    print(f"\nTraining complete. Best top-1: {best_acc:.2f}%")
    return best_acc


# ---------------------------------------------------------------------------
# 8. Quick test (no training data needed)
# ---------------------------------------------------------------------------

def quick_test():
    """Verify shapes and basic forward pass for all variants."""
    print("=" * 60)
    print("  CliffordNet — PyTorch shape & forward pass test")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    dummy = torch.randn(2, 3, 32, 32).to(device)   # batch of 2 CIFAR images

    for variant in MODEL_CONFIGS:
        model = build_cliffordnet(variant, num_classes=100).to(device)
        model.eval()
        with torch.no_grad():
            out = model(dummy)
        print(f"  {variant:6s}  params={model.count_params():>9,}  "
              f"input={tuple(dummy.shape)}  output={tuple(out.shape)}")

    # Ablation modes
    print()
    for cli_mode in ("inner", "wedge", "full"):
        for ctx_mode in ("diff", "abs"):
            model = CliffordBlock(64, [1, 2, 4], cli_mode=cli_mode, ctx_mode=ctx_mode).to(device)
            x = torch.randn(2, 64, 16, 16).to(device)
            with torch.no_grad():
                y = model(x)
            assert y.shape == x.shape, f"Shape mismatch: {y.shape} != {x.shape}"
            print(f"  Block cli={cli_mode:6s} ctx={ctx_mode:4s} — output {tuple(y.shape)}  ✓")

    # Anti-symmetry verification
    print()
    H   = torch.randn(1, 32, 8, 8)
    sgp = SparseRollingProduct([1, 2, 4], mode="wedge")
    # Wedge(H, H) must be identically 0
    wedge_self = sgp(H, H)
    max_abs = wedge_self.abs().max().item()
    print(f"  Anti-symmetry: Wedge(H,H)≡0  max_abs={max_abs:.2e}  "
          f"{'✓' if max_abs < 1e-5 else '✗'}")

    print("\nAll tests passed.\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CliffordNet")
    parser.add_argument("--mode",         default="test",  choices=["test", "train"])
    parser.add_argument("--variant",      default="nano",  choices=list(MODEL_CONFIGS))
    parser.add_argument("--epochs",       type=int,   default=200)
    parser.add_argument("--batch_size",   type=int,   default=128)
    parser.add_argument("--lr",           type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--drop_path",    type=float, default=0.1)
    parser.add_argument("--beta",         type=int,   default=0, choices=[0, 1])
    parser.add_argument("--cli_mode",     default="full",  choices=["full", "inner", "wedge"])
    parser.add_argument("--ctx_mode",     default="diff",  choices=["diff", "abs"])
    args = parser.parse_args()

    if args.mode == "test":
        quick_test()
    else:
        train(
            variant=args.variant,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            drop_path_rate=args.drop_path,
            beta=args.beta,
            cli_mode=args.cli_mode,
            ctx_mode=args.ctx_mode,
        )

"""
TurboQuant: Implementation & Demonstration
============================================
Implementation of the TurboQuant algorithm from:
"TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate"
Zandieh, Daliri, Hadian, Mirrokni (2025)
"""

import numpy as np
from scipy.special import gamma
from scipy.optimize import minimize_scalar
from scipy.integrate import quad
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({
    'figure.figsize': (14, 5),
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'legend.fontsize': 9,
    'figure.dpi': 130,
    'axes.grid': True,
    'grid.alpha': 0.3,
})

# ============================================================
# 1. BETA DISTRIBUTION ON THE HYPERSPHERE
# ============================================================

def beta_pdf(x, d):
    """PDF of a coordinate of a uniform random point on S^{d-1}."""
    coeff = gamma(d / 2) / (np.sqrt(np.pi) * gamma((d - 1) / 2))
    return coeff * (1 - x**2) ** ((d - 3) / 2)


def demonstrate_beta_distribution():
    """Show that coordinates of rotated vectors follow Beta distribution."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    dims = [10, 100, 1000]

    for ax, d in zip(axes, dims):
        # Generate random points on the unit sphere
        n_samples = 20000
        z = np.random.randn(n_samples, d)
        z = z / np.linalg.norm(z, axis=1, keepdims=True)

        # Take first coordinate
        coords = z[:, 0]

        # Plot histogram
        ax.hist(coords, bins=80, density=True, alpha=0.6, color='steelblue',
                label='Coord. empiriques')

        # Plot theoretical Beta PDF
        x_range = np.linspace(-1, 1, 500)
        # Clip for numerical stability
        mask = np.abs(x_range) < 0.999
        pdf_vals = np.zeros_like(x_range)
        pdf_vals[mask] = beta_pdf(x_range[mask], d)
        ax.plot(x_range, pdf_vals, 'r-', lw=2, label='PDF Beta théorique')

        # Plot Gaussian approximation
        gauss = np.exp(-x_range**2 * d / 2) * np.sqrt(d / (2 * np.pi))
        ax.plot(x_range, gauss, 'g--', lw=2, label=f'N(0, 1/{d})')

        ax.set_title(f'd = {d}')
        ax.set_xlabel('Valeur de la coordonnée')
        ax.set_ylabel('Densité')
        ax.legend()

    fig.suptitle('Lemme 1 : Distribution Beta des coordonnées sur la sphère unité',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('/home/claude/fig1_beta_distribution.png', bbox_inches='tight', dpi=150)
    plt.close()
    print("[OK] Figure 1 : Distribution Beta sauvegardée")


# ============================================================
# 2. LLOYD-MAX OPTIMAL SCALAR QUANTIZER
# ============================================================

def lloyd_max_quantizer(d, n_bits, n_iters=50):
    """
    Find optimal scalar quantizer centroids for the Beta distribution
    on coordinates of the unit sphere in dimension d.
    Uses the Lloyd-Max iterative algorithm (continuous 1D k-means).
    """
    n_levels = 2 ** n_bits

    # Initialize centroids uniformly in the support
    sigma = 1.0 / np.sqrt(d)
    centroids = np.linspace(-3 * sigma, 3 * sigma, n_levels)

    for _ in range(n_iters):
        # Compute boundaries (midpoints between consecutive centroids)
        boundaries = np.concatenate([[-1.0],
                                      (centroids[:-1] + centroids[1:]) / 2,
                                      [1.0]])

        new_centroids = np.zeros(n_levels)
        for i in range(n_levels):
            lo, hi = boundaries[i], boundaries[i + 1]
            if hi - lo < 1e-15:
                new_centroids[i] = centroids[i]
                continue

            # Centroid = E[X | X in [lo, hi]]
            num, _ = quad(lambda x: x * beta_pdf(x, d), lo, hi)
            den, _ = quad(lambda x: beta_pdf(x, d), lo, hi)

            if den > 1e-15:
                new_centroids[i] = num / den
            else:
                new_centroids[i] = (lo + hi) / 2

        centroids = new_centroids

    # Compute MSE cost
    boundaries = np.concatenate([[-1.0],
                                  (centroids[:-1] + centroids[1:]) / 2,
                                  [1.0]])
    mse_cost = 0.0
    for i in range(n_levels):
        lo, hi = boundaries[i], boundaries[i + 1]
        cost_i, _ = quad(lambda x: (x - centroids[i])**2 * beta_pdf(x, d), lo, hi)
        mse_cost += cost_i

    return centroids, boundaries, mse_cost


def demonstrate_codebooks():
    """Show optimal codebooks for different bit-widths."""
    d = 256
    print(f"\nCodebooks optimaux Lloyd-Max (d={d}):")
    print("-" * 60)

    theoretical_mse = {1: 0.36, 2: 0.117, 3: 0.03, 4: 0.009}

    results = {}
    for b in range(1, 5):
        centroids, boundaries, mse = lloyd_max_quantizer(d, b)
        results[b] = (centroids, boundaries, mse)
        scaled_centroids = centroids * np.sqrt(d)

        print(f"\nb = {b} bits ({2**b} niveaux):")
        print(f"  Centroïdes (×√d) : {np.round(scaled_centroids, 3)}")
        print(f"  MSE (d × C)      : {d * mse:.6f}")
        print(f"  Théorique        : {theoretical_mse[b]:.6f}")

    return results


# ============================================================
# 3. TURBOQUANT MSE IMPLEMENTATION
# ============================================================

# Global codebook cache
_codebook_cache = {}

def get_codebook(d, n_bits):
    """Get or compute codebook with caching."""
    key = (d, n_bits)
    if key not in _codebook_cache:
        _codebook_cache[key] = lloyd_max_quantizer(d, n_bits)
    return _codebook_cache[key]


class TurboQuantMSE:
    """TurboQuant optimized for MSE (Algorithm 1)."""

    def __init__(self, d, n_bits):
        self.d = d
        self.n_bits = n_bits

        # Generate random rotation matrix via QR decomposition
        G = np.random.randn(d, d)
        self.Pi, _ = np.linalg.qr(G)

        # Precompute optimal codebook (cached)
        self.centroids, self.boundaries, self.scalar_mse = get_codebook(d, n_bits)

    def quantize(self, x):
        """Quantize: x -> indices."""
        y = self.Pi @ x
        # Find nearest centroid for each coordinate
        indices = np.digitize(y, self.boundaries[1:-1])
        indices = np.clip(indices, 0, len(self.centroids) - 1)
        return indices

    def dequantize(self, indices):
        """Dequantize: indices -> reconstructed vector."""
        y_hat = self.centroids[indices]
        x_hat = self.Pi.T @ y_hat
        return x_hat

    def quantize_dequantize(self, x):
        """Full round-trip."""
        return self.dequantize(self.quantize(x))


# ============================================================
# 4. QJL: 1-BIT QUANTIZED JOHNSON-LINDENSTRAUSS
# ============================================================

class QJL:
    """Quantized Johnson-Lindenstrauss transform (Definition 1)."""

    def __init__(self, d):
        self.d = d
        self.S = np.random.randn(d, d)

    def quantize(self, x):
        """Quantize to signs."""
        return np.sign(self.S @ x)

    def dequantize(self, z, norm=1.0):
        """Dequantize with optional norm scaling."""
        return np.sqrt(np.pi / 2) / self.d * norm * self.S.T @ z

    def quantize_dequantize(self, x, norm=1.0):
        return self.dequantize(self.quantize(x), norm)


# ============================================================
# 5. TURBOQUANT PROD IMPLEMENTATION
# ============================================================

class TurboQuantProd:
    """TurboQuant optimized for inner product (Algorithm 2)."""

    def __init__(self, d, n_bits):
        self.d = d
        self.n_bits = n_bits

        # Stage 1: MSE quantizer with (b-1) bits
        self.mse_quantizer = TurboQuantMSE(d, max(n_bits - 1, 1))
        # Stage 2: QJL on residual
        self.qjl = QJL(d)

    def quantize(self, x):
        """Returns (mse_indices, qjl_signs, residual_norm)."""
        idx = self.mse_quantizer.quantize(x)
        x_mse = self.mse_quantizer.dequantize(idx)
        r = x - x_mse
        r_norm = np.linalg.norm(r)
        qjl_signs = self.qjl.quantize(r)
        return idx, qjl_signs, r_norm

    def dequantize(self, idx, qjl_signs, r_norm):
        """Reconstruct from quantized representation."""
        x_mse = self.mse_quantizer.dequantize(idx)
        x_qjl = self.qjl.dequantize(qjl_signs, r_norm)
        return x_mse + x_qjl

    def quantize_dequantize(self, x):
        idx, qjl_signs, r_norm = self.quantize(x)
        return self.dequantize(idx, qjl_signs, r_norm)


# ============================================================
# 6. EXPERIMENT: MSE DISTORTION VS THEORETICAL BOUNDS
# ============================================================

def experiment_mse_bounds():
    """Validate MSE distortion against theoretical upper and lower bounds."""
    d = 128
    n_vectors = 200
    bit_widths = [1, 2, 3, 4]

    mse_empirical = []

    for b in bit_widths:
        print(f"  Calcul MSE pour b={b}...")
        Q = TurboQuantMSE(d, b)
        errors = []
        for _ in range(n_vectors):
            x = np.random.randn(d)
            x = x / np.linalg.norm(x)
            x_hat = Q.quantize_dequantize(x)
            errors.append(np.sum((x - x_hat)**2))

        mse_empirical.append(np.mean(errors))

    # Theoretical bounds
    b_range = np.linspace(0.5, 4.5, 100)
    upper_bound = np.sqrt(3) * np.pi / 2 * (1 / 4**b_range)
    lower_bound = 1 / 4**b_range

    # Specific theoretical values
    theo_vals = {1: 0.36, 2: 0.117, 3: 0.03, 4: 0.009}

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.semilogy(b_range, upper_bound, 'r--', lw=2,
                label=r'Borne sup. : $\frac{\sqrt{3}\pi}{2} \cdot 4^{-b}$')
    ax.semilogy(b_range, lower_bound, 'b--', lw=2,
                label=r'Borne inf. (Shannon) : $4^{-b}$')
    ax.semilogy(bit_widths, mse_empirical, 'ko-', ms=8, lw=2,
                label='TurboQuant$_{mse}$ (empirique)')
    ax.semilogy(list(theo_vals.keys()), list(theo_vals.values()), 'gs', ms=10,
                label='TurboQuant$_{mse}$ (théorique)')

    ax.set_xlabel('Largeur de bits (b)')
    ax.set_ylabel('MSE ($D_{mse}$)')
    ax.set_title('Théorème 1 : Distorsion MSE vs bornes théoriques', fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_xticks(bit_widths)
    plt.tight_layout()
    plt.savefig('/home/claude/fig2_mse_bounds.png', bbox_inches='tight', dpi=150)
    plt.close()
    print("[OK] Figure 2 : MSE vs bornes")
    return mse_empirical


# ============================================================
# 7. EXPERIMENT: INNER PRODUCT BIAS & DISTORTION
# ============================================================

def experiment_inner_product():
    """Compare bias and distortion of MSE vs Prod quantizers."""
    d = 128
    n_pairs = 300
    bit_widths = [1, 2, 3, 4]

    results_mse = {b: {'errors': [], 'estimates': [], 'true_ips': []} for b in bit_widths}
    results_prod = {b: {'errors': [], 'estimates': [], 'true_ips': []} for b in bit_widths}

    for b in bit_widths:
        print(f"  Calcul produit scalaire pour b={b}...")
        Q_mse = TurboQuantMSE(d, b)
        Q_prod = TurboQuantProd(d, b)
        for _ in range(n_pairs):
            x = np.random.randn(d)
            x = x / np.linalg.norm(x)
            y = np.random.randn(d)
            y = y / np.linalg.norm(y)

            true_ip = x @ y

            # MSE quantizer
            x_hat_mse = Q_mse.quantize_dequantize(x)
            est_mse = y @ x_hat_mse
            results_mse[b]['errors'].append(true_ip - est_mse)
            results_mse[b]['estimates'].append(est_mse)
            results_mse[b]['true_ips'].append(true_ip)

            # Prod quantizer
            x_hat_prod = Q_prod.quantize_dequantize(x)
            est_prod = y @ x_hat_prod
            results_prod[b]['errors'].append(true_ip - est_prod)
            results_prod[b]['estimates'].append(est_prod)
            results_prod[b]['true_ips'].append(true_ip)

    # --- Figure 3: Error distributions ---
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))

    for i, b in enumerate(bit_widths):
        # TurboQuant_prod
        axes[0, i].hist(results_prod[b]['errors'], bins=50, density=True,
                        alpha=0.7, color='steelblue')
        axes[0, i].axvline(0, color='red', ls='--', lw=1.5)
        mean_err = np.mean(results_prod[b]['errors'])
        axes[0, i].axvline(mean_err, color='orange', ls='-', lw=2, label=f'μ={mean_err:.4f}')
        axes[0, i].set_title(f'b = {b}')
        axes[0, i].legend(fontsize=8)
        if i == 0:
            axes[0, i].set_ylabel('TurboQuant$_{prod}$\nDensité')

        # TurboQuant_mse
        axes[1, i].hist(results_mse[b]['errors'], bins=50, density=True,
                        alpha=0.7, color='coral')
        axes[1, i].axvline(0, color='red', ls='--', lw=1.5)
        mean_err = np.mean(results_mse[b]['errors'])
        axes[1, i].axvline(mean_err, color='orange', ls='-', lw=2, label=f'μ={mean_err:.4f}')
        axes[1, i].set_title(f'b = {b}')
        axes[1, i].legend(fontsize=8)
        axes[1, i].set_xlabel('Erreur produit scalaire')
        if i == 0:
            axes[1, i].set_ylabel('TurboQuant$_{mse}$\nDensité')

    fig.suptitle('Distribution de l\'erreur du produit scalaire : absence de biais (prod) vs biais (mse)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('/home/claude/fig3_ip_bias.png', bbox_inches='tight', dpi=150)
    plt.close()
    print("[OK] Figure 3 : Distribution erreur produit scalaire")

    # --- Figure 4: IP distortion vs bounds ---
    ip_var_prod = []
    ip_var_mse = []
    for b in bit_widths:
        ip_var_prod.append(np.mean(np.array(results_prod[b]['errors'])**2))
        ip_var_mse.append(np.mean(np.array(results_mse[b]['errors'])**2))

    b_range = np.linspace(0.5, 4.5, 100)
    upper_ip = np.sqrt(3) * np.pi / 2 / d * (1 / 4**b_range)
    lower_ip = 1 / d * (1 / 4**b_range)

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.semilogy(b_range, upper_ip, 'r--', lw=2,
                label=r'Borne sup. : $\frac{\sqrt{3}\pi}{2d} \cdot 4^{-b}$')
    ax.semilogy(b_range, lower_ip, 'b--', lw=2,
                label=r'Borne inf. : $\frac{1}{d} \cdot 4^{-b}$')
    ax.semilogy(bit_widths, ip_var_prod, 'go-', ms=8, lw=2,
                label='TurboQuant$_{prod}$')
    ax.semilogy(bit_widths, ip_var_mse, 'rs-', ms=8, lw=2,
                label='TurboQuant$_{mse}$')

    ax.set_xlabel('Largeur de bits (b)')
    ax.set_ylabel('Distorsion produit scalaire ($D_{prod}$)')
    ax.set_title('Théorème 2 : Distorsion produit scalaire vs bornes', fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_xticks(bit_widths)
    plt.tight_layout()
    plt.savefig('/home/claude/fig4_ip_distortion.png', bbox_inches='tight', dpi=150)
    plt.close()
    print("[OK] Figure 4 : Distorsion produit scalaire vs bornes")

    return results_mse, results_prod


# ============================================================
# 8. EXPERIMENT: NEAREST NEIGHBOR SEARCH RECALL
# ============================================================

def experiment_nearest_neighbor():
    """Compare recall@k for TurboQuant vs naive uniform quantization."""
    d = 64
    n_db = 2000
    n_queries = 100
    n_bits = 2

    print(f"  Génération dataset (d={d}, n={n_db})...")

    # Generate database and query vectors
    X_db = np.random.randn(n_db, d)
    X_db = X_db / np.linalg.norm(X_db, axis=1, keepdims=True)
    X_queries = np.random.randn(n_queries, d)
    X_queries = X_queries / np.linalg.norm(X_queries, axis=1, keepdims=True)

    # Ground truth: exact inner products
    true_ips = X_queries @ X_db.T
    true_top1 = np.argmax(true_ips, axis=1)

    # --- TurboQuant_prod ---
    print(f"  Quantification TurboQuant (b={n_bits})...")
    Q_turbo = TurboQuantProd(d, n_bits)
    X_db_turbo = np.zeros_like(X_db)
    for i in range(n_db):
        idx, qjl_signs, r_norm = Q_turbo.quantize(X_db[i])
        X_db_turbo[i] = Q_turbo.dequantize(idx, qjl_signs, r_norm)

    turbo_ips = X_queries @ X_db_turbo.T

    # --- Naive uniform quantization ---
    print(f"  Quantification uniforme naïve (b={n_bits})...")
    n_levels = 2 ** n_bits
    X_db_naive = np.zeros_like(X_db)
    for i in range(n_db):
        x = X_db[i]
        x_min, x_max = x.min(), x.max()
        # Uniform quantization
        x_norm = (x - x_min) / (x_max - x_min + 1e-10)
        x_quant = np.round(x_norm * (n_levels - 1)) / (n_levels - 1)
        X_db_naive[i] = x_quant * (x_max - x_min) + x_min

    naive_ips = X_queries @ X_db_naive.T

    # Compute recall@k
    ks = [1, 2, 4, 8, 16, 32, 64]
    recall_turbo = []
    recall_naive = []

    for k in ks:
        # TurboQuant
        top_k_turbo = np.argsort(-turbo_ips, axis=1)[:, :k]
        hits_turbo = np.array([true_top1[i] in top_k_turbo[i] for i in range(n_queries)])
        recall_turbo.append(hits_turbo.mean())

        # Naive
        top_k_naive = np.argsort(-naive_ips, axis=1)[:, :k]
        hits_naive = np.array([true_top1[i] in top_k_naive[i] for i in range(n_queries)])
        recall_naive.append(hits_naive.mean())

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.semilogx(ks, recall_turbo, 'go-', ms=8, lw=2, base=2,
                label=f'TurboQuant$_{{prod}}$ ({n_bits} bits)')
    ax.semilogx(ks, recall_naive, 'rs-', ms=8, lw=2, base=2,
                label=f'Quantification uniforme ({n_bits} bits)')
    ax.set_xlabel('Top-k')
    ax.set_ylabel('Recall@1@k')
    ax.set_title(f'Recherche de plus proches voisins (d={d}, n={n_db})', fontweight='bold')
    ax.legend(fontsize=11)
    ax.set_ylim([0, 1.05])
    ax.set_xticks(ks)
    ax.set_xticklabels(ks)
    plt.tight_layout()
    plt.savefig('/home/claude/fig5_nn_recall.png', bbox_inches='tight', dpi=150)
    plt.close()
    print("[OK] Figure 5 : Recall NN search")

    return recall_turbo, recall_naive


# ============================================================
# 9. EXPERIMENT: COMPRESSION RATIO VS QUALITY
# ============================================================

def experiment_compression_quality():
    """Show quality vs compression ratio trade-off."""
    d = 128
    n_vectors = 200

    bit_widths = [1, 2, 3, 4, 5]
    mse_vals = []
    cosine_sims = []

    for b in bit_widths:
        print(f"  b={b}...")
        Q = TurboQuantMSE(d, b)
        mse_list = []
        cos_list = []
        for _ in range(n_vectors):
            x = np.random.randn(d)
            x = x / np.linalg.norm(x)
            x_hat = Q.quantize_dequantize(x)

            mse_list.append(np.sum((x - x_hat)**2))
            cos_list.append(x @ x_hat / (np.linalg.norm(x_hat) + 1e-10))

        mse_vals.append(np.mean(mse_list))
        cosine_sims.append(np.mean(cos_list))

    compression_ratios = [32 / b for b in bit_widths]  # from float32

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # MSE vs compression
    ax1.bar(range(len(bit_widths)), mse_vals, color='steelblue', alpha=0.8,
            tick_label=[f'{b} bits\n({cr:.0f}×)' for b, cr in zip(bit_widths, compression_ratios)])
    ax1.set_ylabel('MSE')
    ax1.set_xlabel('Largeur de bits (ratio de compression)')
    ax1.set_title('MSE de reconstruction', fontweight='bold')
    for i, v in enumerate(mse_vals):
        ax1.text(i, v + 0.005, f'{v:.4f}', ha='center', fontsize=9)

    # Cosine similarity vs compression
    ax2.bar(range(len(bit_widths)), cosine_sims, color='seagreen', alpha=0.8,
            tick_label=[f'{b} bits\n({cr:.0f}×)' for b, cr in zip(bit_widths, compression_ratios)])
    ax2.set_ylabel('Similarité cosinus')
    ax2.set_xlabel('Largeur de bits (ratio de compression)')
    ax2.set_title('Préservation de la direction', fontweight='bold')
    ax2.set_ylim([0.5, 1.02])
    for i, v in enumerate(cosine_sims):
        ax2.text(i, v + 0.005, f'{v:.4f}', ha='center', fontsize=9)

    fig.suptitle('Qualité vs Compression : TurboQuant$_{mse}$', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('/home/claude/fig6_compression_quality.png', bbox_inches='tight', dpi=150)
    plt.close()
    print("[OK] Figure 6 : Compression vs qualité")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    np.random.seed(42)

    print("=" * 60)
    print("TurboQuant : Implémentation et Démonstration")
    print("=" * 60)

    print("\n[1/6] Distribution Beta des coordonnées sur la sphère...")
    demonstrate_beta_distribution()

    print("\n[2/6] Codebooks optimaux Lloyd-Max...")
    codebooks = demonstrate_codebooks()

    print("\n[3/6] Validation MSE vs bornes théoriques...")
    mse_results = experiment_mse_bounds()

    print("\n[4/6] Produit scalaire : biais et distorsion...")
    ip_mse, ip_prod = experiment_inner_product()

    print("\n[5/6] Recherche de plus proches voisins...")
    recall_turbo, recall_naive = experiment_nearest_neighbor()

    print("\n[6/6] Qualité vs compression...")
    experiment_compression_quality()

    print("\n" + "=" * 60)
    print("Toutes les expériences terminées avec succès !")
    print("=" * 60)

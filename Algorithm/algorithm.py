import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from scipy.optimize import minimize
import matplotlib.animation as animation
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)


class GombocProteinModel:
    def __init__(self, n_points=48, ruggedness=0.00):
        self.n_points = n_points
        # Scale ruggedness appropriately - now 0.01 means 1% of energy scale
        self.ruggedness = ruggedness * 100.0  # Scale up because energy terms are ~100-300
        self.native_shape = self._create_gomboc_like_shape()
        self.folding_pathways = []
        self.energy_min = float('inf')
        self.energy_max = float('-inf')
        self.all_results = []
        self.native_energy = self._energy_fun(self.native_shape.flatten())

    def _create_gomboc_like_shape(self):
        theta = np.linspace(0, 2 * np.pi, self.n_points, endpoint=False)
        r = 1.0 + 0.20 * np.sin(theta) + 0.12 * np.sin(2.8 * theta + 0.5) + 0.08 * np.cos(4.5 * theta)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        points = np.column_stack([x, y])
        points[:, 0] += 0.12 * points[:, 1] ** 2 - 0.04 * points[:, 1] ** 3
        points = points / np.max(np.abs(points)) * 1.5
        return self._sort_by_angle(points)

    def _sort_by_angle(self, points):
        c = np.mean(points, axis=0)
        angles = np.arctan2(points[:, 1] - c[1], points[:, 0] - c[0])
        return points[np.argsort(angles)]

    def _raw_rmsd(self, P, Q):
        return np.sqrt(np.mean(np.sum((P - Q) ** 2, axis=1)))

    def _energy_fun(self, flat_shape):
        shape = flat_shape.reshape(-1, 2)
        dev = self._raw_rmsd(shape, self.native_shape)

        # Native drive - main folding force
        if dev < 0.3:
            native_drive = 150.0 * dev ** 2
        else:
            native_drive = 90.0 * dev - 13.5

        # Confinement - keeps shape from expanding too much
        r2 = np.mean(np.sum(shape ** 2, axis=1))
        confinement = 8.0 * max(0, r2 - 2.5) ** 2

        # Repulsion - prevents self-intersection
        repulsion = 0.0
        dmat = np.linalg.norm(shape[:, None] - shape[None, :], axis=-1)
        close_pairs = []
        for i in range(self.n_points):
            for j in range(i + 2, self.n_points):
                if j != i + 1 and j != i - 1 and j != i and (j + 1) % self.n_points != i:
                    dist = dmat[i, j]
                    if dist < 0.25:
                        close_pairs.append((i, j, dist))

        if close_pairs:
            repulsion = sum(np.exp(8.0 * (0.25 - dist)) for _, _, dist in close_pairs) * 0.5

        # Local structure - maintains reasonable angles
        local_structure = 0.0
        for i in range(self.n_points):
            j = (i + 1) % self.n_points
            k = (i + 2) % self.n_points
            v1 = shape[j] - shape[i]
            v2 = shape[k] - shape[j]
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            if norm1 > 1e-6 and norm2 > 1e-6:
                cos_angle = np.dot(v1, v2) / (norm1 * norm2)
                cos_angle = np.clip(cos_angle, -0.9999, 0.9999)
                angle = np.arccos(cos_angle)
                local_structure += (angle - 2.0) ** 2 * 2.0

        # Bond lengths - maintains consistent segment lengths
        bond_lengths = 0.0
        for i in range(self.n_points):
            j = (i + 1) % self.n_points
            bond_len = np.linalg.norm(shape[j] - shape[i])
            bond_lengths += (bond_len - 0.22) ** 2 * 5.0

        # RUGGEDNESS TERMS - properly scaled
        r = self.ruggedness

        # Random roughness - creates local energy variations
        if r > 0:
            roughness = r * 0.01 * np.sum(np.random.randn(*shape.ravel().shape) *
                                          np.sin(20.0 * shape.ravel()))
        else:
            roughness = 0.0

        # Frustration - long-range conflicts that create traps
        frustration = 0.0
        step = 5
        idx = np.arange(0, self.n_points, step)
        n_idx = len(idx)
        if n_idx >= 5 and r > 0:
            i, j = np.triu_indices(n_idx, k=2)
            nat_d = np.linalg.norm(self.native_shape[idx][i] - self.native_shape[idx][j], axis=1)
            cur_d = np.linalg.norm(shape[idx][i] - shape[idx][j], axis=1)
            delta = cur_d - nat_d
            mask = np.abs(delta) > 0.2
            if np.any(mask):
                # Scale frustration to be comparable to native_drive
                frustration = r * 0.1 * np.mean(delta[mask] ** 2) * 150.0

        total = native_drive + confinement + repulsion + local_structure + bond_lengths + roughness + frustration

        # Update min/max tracking for visualization
        self.energy_min = min(self.energy_min, total)
        self.energy_max = max(self.energy_max, total)

        return total

    def perturb_shape(self, strength=0.3):
        noise = np.random.randn(self.n_points, 2) * strength

        # Add correlated noise for more realistic perturbations
        for i in range(0, self.n_points, 8):
            if i + 4 < self.n_points:
                direction = np.random.randn(2)
                noise[i:i + 4] += direction * 0.1

        p = self.native_shape + noise

        # Keep within bounds
        max_abs = np.max(np.abs(p))
        if max_abs > 1.8:
            p = p / max_abs * 1.7

        return self._sort_by_angle(p)

    def simulate_folding(self, maxiter=8000):
        # Random initial perturbation strength
        choice = np.random.rand()
        if choice < 0.3:
            strength = 0.2
        elif choice < 0.6:
            strength = 0.35
        else:
            strength = 0.5

        initial_flat = self.perturb_shape(strength=strength).flatten()

        path = [initial_flat.reshape(-1, 2).copy()]
        energies = [self._energy_fun(initial_flat)]

        def callback(xk):
            path.append(xk.reshape(-1, 2).copy())
            energies.append(self._energy_fun(xk))

        # Two-stage optimization: CG then BFGS
        res1 = minimize(
            self._energy_fun,
            initial_flat,
            method='CG',
            options={'maxiter': 2000, 'gtol': 1e-3},
            callback=callback
        )

        res2 = minimize(
            self._energy_fun,
            res1.x,
            method='BFGS',
            options={'maxiter': 6000, 'gtol': 1e-6},
            callback=callback
        )

        final_flat = res2.x
        final_shape = final_flat.reshape(-1, 2)
        final_energy = self._energy_fun(final_flat)

        self.folding_pathways.append({'pathway': path, 'energies': energies})

        rmsd_final = self._raw_rmsd(final_shape, self.native_shape)
        self.all_results.append({
            'initial_e': energies[0],
            'final_e': final_energy,
            'rmsd': rmsd_final,
            'steps': len(path) - 1
        })

        return path, energies, final_energy, final_shape

    def run_multiple_folding_experiments(self, n_experiments=30, maxiter=8000):
        print("\n" + "═" * 80)
        print(f"  MULTIPLE FOLDING SIMULATIONS  –  ruggedness = {self.ruggedness / 100.0:.3f}")
        print("═" * 80)

        for i in range(n_experiments):
            path, energies, final_e, final_shape = self.simulate_folding(maxiter=maxiter)
            init_e = energies[0]
            rmsd_final = self._raw_rmsd(final_shape, self.native_shape)
            n_steps = len(path) - 1

            if rmsd_final < 0.12:
                status = "🏆 PERFECT"
            elif rmsd_final < 0.20:
                status = "✅ SUCCESS"
            elif rmsd_final < 0.30:
                status = "👍 GOOD"
            elif rmsd_final < 0.40:
                status = "🔄 NEAR-NATIVE"
            elif rmsd_final < 0.55:
                status = "⚠️ MISFOLDED"
            else:
                status = "❌ POOR"

            # Calculate energy change to show folding progress
            energy_change = init_e - final_e
            change_symbol = "↓" if energy_change > 10 else "→" if abs(energy_change) <= 10 else "↑"

            print(
                f"Run {i + 1:2d} | {init_e:5.1f} {change_symbol} {final_e:5.1f}   "
                f"RMSD {rmsd_final:.3f}   steps {n_steps:3d}   {status}")

        self._print_statistics()

    def _print_statistics(self):
        print("\n" + "─" * 80)
        print("SUMMARY STATISTICS")

        rmsds = [r['rmsd'] for r in self.all_results]
        init_es = [r['initial_e'] for r in self.all_results]
        final_es = [r['final_e'] for r in self.all_results]
        steps = [r['steps'] for r in self.all_results]
        energy_drops = [i - f for i, f in zip(init_es, final_es)]

        print(f"Native energy:       {self.native_energy:5.1f}")
        print(f"Avg initial E:       {np.mean(init_es):5.1f} ± {np.std(init_es):.1f}")
        print(f"Avg final E:         {np.mean(final_es):5.1f} ± {np.std(final_es):.1f}")
        print(f"Avg energy drop:     {np.mean(energy_drops):5.1f} ± {np.std(energy_drops):.1f}")
        print(f"Avg final RMSD:      {np.mean(rmsds):.3f} ± {np.std(rmsds):.3f}")
        print(f"Avg steps:           {np.mean(steps):.0f}")

        perfect = 100 * np.mean(np.array(rmsds) <= 0.12)
        success = 100 * np.mean((np.array(rmsds) > 0.12) & (np.array(rmsds) <= 0.20))
        good = 100 * np.mean((np.array(rmsds) > 0.20) & (np.array(rmsds) <= 0.30))
        near = 100 * np.mean((np.array(rmsds) > 0.30) & (np.array(rmsds) <= 0.40))
        misfolded = 100 * np.mean((np.array(rmsds) > 0.40) & (np.array(rmsds) <= 0.55))
        poor = 100 - perfect - success - good - near - misfolded

        print(f"\nOutcome distribution:")
        print(f"  • 🏆 Perfect      (RMSD ≤ 0.12) : {perfect:5.1f}%")
        print(f"  • ✅ Successful   (≤ 0.20)     : {success:5.1f}%")
        print(f"  • 👍 Good         (≤ 0.30)     : {good:5.1f}%")
        print(f"  • 🔄 Near-native  (≤ 0.40)     : {near:5.1f}%")
        print(f"  • ⚠️  Misfolded    (≤ 0.55)     : {misfolded:5.1f}%")
        print(f"  • ❌ Poor         (> 0.55)      : {poor:5.1f}%")

    def _get_color(self, energy):
        if self.energy_max == self.energy_min:
            return (0.5, 0.5, 0.5)
        norm = (energy - self.energy_min) / (self.energy_max - self.energy_min)
        norm = np.clip(norm, 0, 1)
        return (norm, 0.3 + 0.7 * (1 - norm), 1.0 - 0.7 * norm)

    def create_animation(self, filename="gomboc_folding_improved.gif"):
        if not self.folding_pathways:
            print("No pathway available.")
            return

        # Find best folding pathway (largest energy drop)
        improvements = []
        for p in self.folding_pathways:
            if len(p['energies']) > 1:
                improvement = p['energies'][0] - p['energies'][-1]
                improvements.append(improvement)
            else:
                improvements.append(0)

        if max(improvements) > 0:
            best_idx = np.argmax(improvements)
        else:
            # If no improvement, use smallest RMSD
            rmsds_final = [self._raw_rmsd(p['pathway'][-1], self.native_shape)
                           for p in self.folding_pathways]
            best_idx = np.argmin(rmsds_final)

        pathway = self.folding_pathways[best_idx]['pathway']
        energies = self.folding_pathways[best_idx]['energies']
        rmsds = [self._raw_rmsd(p, self.native_shape) for p in pathway]

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.5, 5))

        def animate(frame):
            ax1.clear()
            ax2.clear()
            ax3.clear()

            shape = pathway[frame]
            cshape = shape - np.mean(shape, axis=0)
            ax1.add_patch(Polygon(cshape, closed=True,
                                  facecolor=self._get_color(energies[frame]),
                                  edgecolor='k', lw=1.2, alpha=0.93))
            cn = self.native_shape - np.mean(self.native_shape, axis=0)
            ax1.add_patch(Polygon(cn, closed=True, facecolor='none',
                                  edgecolor='navy', lw=2.3, ls='--', alpha=0.7))
            ax1.set_xlim(-1.8, 1.8)
            ax1.set_ylim(-1.8, 1.8)
            ax1.set_aspect('equal')
            ax1.axis('off')
            ax1.set_title(f"Step {frame:3d}   E = {energies[frame]:.1f}   RMSD {rmsds[frame]:.3f}")

            ax2.plot(energies[:frame + 1], 'b-', lw=2.1)
            ax2.axhline(self.native_energy, color='r', ls='--', lw=1.8, label='native')
            ax2.set_xlim(0, max(100, len(energies) - 1))
            if len(energies[:frame + 1]) > 0:
                y_min = min(min(energies[:frame + 1]), self.native_energy) - 3
                y_max = max(energies[:frame + 1]) + 5
                ax2.set_ylim(y_min, y_max)
            ax2.grid(alpha=0.3)
            ax2.set_xlabel("step")
            ax2.set_ylabel("energy")
            if frame == 0:
                ax2.legend()

            ax3.plot(rmsds[:frame + 1], 'g-', lw=2.1)
            ax3.axhline(0.12, color='green', ls='--', alpha=0.8, label='perfect')
            ax3.axhline(0.20, color='blue', ls='--', alpha=0.8, label='success')
            ax3.axhline(0.30, color='cyan', ls='--', alpha=0.8, label='good')
            ax3.axhline(0.40, color='orange', ls='--', alpha=0.8, label='near')
            ax3.axhline(0.55, color='red', ls='--', alpha=0.8, label='misfolded')
            ax3.set_xlim(0, max(100, len(rmsds) - 1))
            ax3.set_ylim(0, 1.0)
            ax3.grid(alpha=0.3)
            ax3.set_xlabel("step")
            ax3.set_ylabel("RMSD")
            if frame == 0:
                ax3.legend(fontsize=7)

        anim = animation.FuncAnimation(fig, animate, frames=len(pathway),
                                       interval=70, repeat=True)

        print(f"\nSaving animation → {filename}")
        anim.save(filename, writer='pillow', fps=10, dpi=120)
        plt.close(fig)
        print("Animation saved.\n")


def visualize_final_results(model):
    """Create comprehensive visualization of 30 runs"""
    if not model.all_results:
        print("No results to visualize")
        return

    # Extract data
    rmsds = [r['rmsd'] for r in model.all_results]
    final_es = [r['final_e'] for r in model.all_results]
    init_es = [r['initial_e'] for r in model.all_results]
    steps = [r['steps'] for r in model.all_results]
    energy_drops = [i - f for i, f in zip(init_es, final_es)]

    # Calculate statistics
    near_native_pct = 100 * np.mean(np.array(rmsds) <= 0.40)
    good_pct = 100 * np.mean(np.array(rmsds) <= 0.30)
    avg_energy_drop = np.mean(energy_drops)

    # Create figure
    fig = plt.figure(figsize=(18, 10))

    # 1. Main scatter plot (Energy vs RMSD)
    ax1 = plt.subplot(2, 3, 1)
    categories = []
    for rmsd in rmsds:
        if rmsd <= 0.20:
            categories.append('green')
        elif rmsd <= 0.30:
            categories.append('blue')
        elif rmsd <= 0.40:
            categories.append('orange')
        elif rmsd <= 0.55:
            categories.append('red')
        else:
            categories.append('darkred')

    scatter = ax1.scatter(final_es, rmsds, c=categories, s=100, alpha=0.7,
                          edgecolors='black', linewidth=1)
    ax1.set_xlabel('Final Energy', fontsize=12)
    ax1.set_ylabel('Final RMSD', fontsize=12)
    ax1.set_title(f'Energy-RMSD Relationship\n({near_native_pct:.1f}% reach near-native)', fontsize=14)
    ax1.grid(True, alpha=0.3)

    # Add trend line if enough points
    if len(final_es) > 1:
        z = np.polyfit(final_es, rmsds, 1)
        p = np.poly1d(z)
        x_trend = np.linspace(min(final_es), max(final_es), 100)
        ax1.plot(x_trend, p(x_trend), 'r--', linewidth=2,
                 label=f'Correlation: {np.corrcoef(final_es, rmsds)[0, 1]:.3f}')
        ax1.legend()

    # 2. RMSD Distribution
    ax2 = plt.subplot(2, 3, 2)
    n, bins, patches = ax2.hist(rmsds, bins=15, edgecolor='black', alpha=0.7)

    # Color bars by category
    for i, patch in enumerate(patches):
        if bins[i] <= 0.20:
            patch.set_facecolor('green')
        elif bins[i] <= 0.30:
            patch.set_facecolor('blue')
        elif bins[i] <= 0.40:
            patch.set_facecolor('orange')
        elif bins[i] <= 0.55:
            patch.set_facecolor('red')
        else:
            patch.set_facecolor('darkred')

    ax2.axvline(np.mean(rmsds), color='black', linestyle='--', linewidth=2,
                label=f'Mean: {np.mean(rmsds):.3f}')
    ax2.axvline(0.20, color='green', linestyle=':', linewidth=1.5, label='Success (0.20)')
    ax2.axvline(0.30, color='blue', linestyle=':', linewidth=1.5, label='Good (0.30)')
    ax2.axvline(0.40, color='orange', linestyle=':', linewidth=1.5, label='Near-native (0.40)')
    ax2.set_xlabel('RMSD', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title(f'RMSD Distribution\n{good_pct:.1f}% ≤ 0.30', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Pie chart of outcomes
    ax3 = plt.subplot(2, 3, 3)

    # Calculate all categories
    perfect_count = sum(np.array(rmsds) <= 0.12)
    success_count = sum((np.array(rmsds) > 0.12) & (np.array(rmsds) <= 0.20))
    good_count = sum((np.array(rmsds) > 0.20) & (np.array(rmsds) <= 0.30))
    near_count = sum((np.array(rmsds) > 0.30) & (np.array(rmsds) <= 0.40))
    misfolded_count = sum((np.array(rmsds) > 0.40) & (np.array(rmsds) <= 0.55))
    poor_count = sum(np.array(rmsds) > 0.55)

    sizes = [perfect_count, success_count, good_count, near_count, misfolded_count, poor_count]
    colors = ['darkgreen', 'green', 'blue', 'orange', 'red', 'darkred']
    labels = [f'Perfect (≤0.12): {sizes[0]}',
              f'Success (≤0.20): {sizes[1]}',
              f'Good (≤0.30): {sizes[2]}',
              f'Near-native (≤0.40): {sizes[3]}',
              f'Misfolded (≤0.55): {sizes[4]}',
              f'Poor (>0.55): {sizes[5]}']

    # Only show non-zero categories
    non_zero = [(s, c, l) for s, c, l in zip(sizes, colors, labels) if s > 0]
    if non_zero:
        sizes_filt, colors_filt, labels_filt = zip(*non_zero)
        wedges, texts, autotexts = ax3.pie(sizes_filt, colors=colors_filt, autopct='%1.1f%%',
                                           startangle=90, explode=[0.05] * len(sizes_filt))
        ax3.legend(wedges, labels_filt, loc="center left", bbox_to_anchor=(1, 0, 0.5, 1), fontsize=9)
    ax3.set_title(f'Folding Outcomes\n{near_native_pct:.1f}% Near-native or better', fontsize=14)

    # 4. Steps vs RMSD
    ax4 = plt.subplot(2, 3, 4)
    scatter4 = ax4.scatter(steps, rmsds, c=energy_drops, cmap='viridis', s=100, alpha=0.7,
                           edgecolors='black', linewidth=1)
    ax4.set_xlabel('Number of Steps', fontsize=12)
    ax4.set_ylabel('Final RMSD', fontsize=12)
    ax4.set_title('Folding Speed vs Quality', fontsize=14)
    plt.colorbar(scatter4, ax=ax4, label='Energy Drop')
    ax4.grid(True, alpha=0.3)

    # 5. Initial vs Final Energy
    ax5 = plt.subplot(2, 3, 5)
    scatter5 = ax5.scatter(init_es, final_es, c=rmsds, cmap='viridis_r',
                           s=100, alpha=0.7, edgecolors='black', linewidth=1)
    ax5.plot([min(init_es), max(init_es)], [min(init_es), max(init_es)],
             'k--', alpha=0.5, label='No improvement')
    ax5.set_xlabel('Initial Energy', fontsize=12)
    ax5.set_ylabel('Final Energy', fontsize=12)
    ax5.set_title(f'Energy Minimization\nAvg Drop: {avg_energy_drop:.1f}', fontsize=14)
    plt.colorbar(scatter5, ax=ax5, label='RMSD')
    ax5.grid(True, alpha=0.3)

    # 6. Cumulative success curve
    ax6 = plt.subplot(2, 3, 6)
    sorted_rmsds = np.sort(rmsds)
    cumulative = np.arange(1, len(sorted_rmsds) + 1) / len(sorted_rmsds) * 100

    ax6.plot(sorted_rmsds, cumulative, 'b-', linewidth=3, marker='o', markersize=8)
    ax6.axhline(50, color='gray', linestyle='--', alpha=0.7, label='50%')
    ax6.axhline(90, color='gray', linestyle='--', alpha=0.7, label='90%')
    ax6.axvline(0.30, color='green', linestyle='--', linewidth=2, label='Good (0.30)')
    ax6.axvline(0.40, color='orange', linestyle='--', linewidth=2, label='Near-native (0.40)')
    ax6.set_xlabel('RMSD', fontsize=12)
    ax6.set_ylabel('Cumulative Percentage (%)', fontsize=12)

    # Find 90th percentile
    pct_90_idx = int(0.9 * len(sorted_rmsds)) - 1
    pct_90_rmsd = sorted_rmsds[max(0, pct_90_idx)]
    ax6.set_title(f'Cumulative Success Curve\n90% below {pct_90_rmsd:.2f}', fontsize=14)
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('gomboc_final_results.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Print statistics
    print("\n" + "=" * 60)
    print("🎯 KEY ACHIEVEMENTS")
    print("=" * 60)
    print(f"✓ {near_native_pct:.1f}% of runs reach near-native or better (RMSD ≤ 0.40)")
    print(f"✓ {good_pct:.1f}% reach good or better (RMSD ≤ 0.30)")
    print(f"✓ {100 * np.mean(np.array(rmsds) > 0.55):.1f}% completely fail (poor folds)")
    print(f"✓ Best RMSD: {min(rmsds):.3f} (Run {np.argmin(rmsds) + 1})")
    print(f"✓ Average RMSD: {np.mean(rmsds):.3f} ± {np.std(rmsds):.3f}")
    print(f"✓ Average energy drop: {avg_energy_drop:.1f} units")
    print("=" * 60)


def main():
    print("=" * 90)
    print("   Gömböc – Protein Folding Analogy     (corrected version)")
    print("=" * 90)

    # Create model - try different ruggedness values:
    model = GombocProteinModel(n_points=48, ruggedness=0.00)
    print(f"Native energy: {model.native_energy:.1f}\n")

    # Run simulations
    model.run_multiple_folding_experiments(n_experiments=100, maxiter=10000)

    # Create animation
    print("\nGenerating animation of best trajectory...")
    model.create_animation("gomboc_folding_improved.gif")

    # Create visualization
    print("\nGenerating final results visualization...")
    visualize_final_results(model)

    print("\nDone.")


if __name__ == "__main__":
    main()

"""
Generate comparison figures for FLUXVortex vs PteraSoftware README.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import os

fig_dir = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(fig_dir, exist_ok=True)

# ═══════════════════════════════════════════════════════════════
# Figure 1: Accuracy comparison — bar chart
# ═══════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor='white')

# --- Panel A: Static Wing CL ---
ax = axes[0]
methods = ['XFLR5\n(reference)', 'PteraSoftware\n(ring-wake)', 'FLUXVortex\nN=5', 'FLUXVortex\nN=10', 'FLUXVortex\nN=20']
cl_vals = [0.485, 0.5180, 0.4635, 0.5020, 0.5180]
cl_errs = [0, 6.8, 4.4, 3.5, 6.8]
colors = ['#888888', '#4a90d9', '#e8734a', '#2ca02c', '#984ea3']

bars = ax.bar(methods, cl_vals, color=colors, edgecolor='#333333', linewidth=0.8, width=0.6)
ax.axhline(y=0.485, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='XFLR5 Reference')
ax.set_ylabel('CL (Lift Coefficient)', fontsize=11)
ax.set_title('Case 1: Static Wing CL\n(NACA 2412, α=5°, AR=10)', fontsize=12, fontweight='bold')
ax.set_ylim(0.40, 0.56)
ax.legend(fontsize=9, loc='upper left')

for bar, err in zip(bars, cl_errs):
    height = bar.get_height()
    if err > 0:
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.003,
                f'{height:.4f}\n(err={err:.1f}%)', ha='center', va='bottom', fontsize=8)
    else:
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.003,
                f'{height:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

ax.grid(axis='y', alpha=0.3)

# --- Panel B: Plunging Wing vs Theodorsen ---
ax = axes[1]
ks = [0.5, 0.2, 0.1]
x = np.arange(len(ks))
width = 0.25

ring_ratios = [0.929, 0.930, 0.920]
hybrid10_ratios = [0.974, 0.946, 0.925]
hybrid20_ratios = [0.930, 0.932, 0.920]

bars1 = ax.bar(x - width, ring_ratios, width, label='PteraSoftware\n(ring-wake)',
               color='#4a90d9', edgecolor='#333', linewidth=0.8)
bars2 = ax.bar(x, hybrid10_ratios, width, label='FLUXVortex\nN=10 FREE',
               color='#2ca02c', edgecolor='#333', linewidth=0.8)
bars3 = ax.bar(x + width, hybrid20_ratios, width, label='FLUXVortex\nN=20 FREE',
               color='#984ea3', edgecolor='#333', linewidth=0.8)

ax.axhline(y=1.0, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Theodorsen (exact)')
ax.set_xticks(x)
ax.set_xticklabels([f'k={k}' for k in ks], fontsize=10)
ax.set_ylabel('CL Amplitude / Theodorsen', fontsize=11)
ax.set_title('Case 2: Plunging Wing Accuracy\n(NACA 0012, AR=10, h₀/c=0.1)', fontsize=12, fontweight='bold')
ax.set_ylim(0.85, 1.05)
ax.legend(fontsize=9, loc='lower right')

for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.003,
                f'{height:.3f}', ha='center', va='bottom', fontsize=8)

ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
path = os.path.join(fig_dir, 'accuracy_comparison.png')
fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
print(f"Saved: {path}")
plt.close(fig)


# ═══════════════════════════════════════════════════════════════
# Figure 2: Radar / summary comparison
# ═══════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')

# Accuracy comparison table-style figure
categories = [
    'Static CL\nvs XFLR5',
    'Plunging k=0.5\nvs Theodorsen',
    'Plunging k=0.2\nvs Theodorsen',
    'Plunging k=0.1\nvs Theodorsen',
]

ptera_data = [93.2, 92.9, 93.0, 92.0]   # % accuracy (100 - err% or ratio*100)
flux_data = [96.5, 97.4, 94.6, 92.5]    # N=10 FREE

x = np.arange(len(categories))
width = 0.35

bars1 = ax.bar(x - width/2, ptera_data, width, label='PteraSoftware (ring-wake)',
               color='#4a90d9', edgecolor='#333', linewidth=0.8)
bars2 = ax.bar(x + width/2, flux_data, width, label='FLUXVortex (Hybrid N=10 FREE)',
               color='#2ca02c', edgecolor='#333', linewidth=0.8)

ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=10)
ax.set_ylabel('Accuracy (% of reference)', fontsize=11)
ax.set_title('FLUXVortex vs PteraSoftware: Accuracy Across All Test Cases', fontsize=13, fontweight='bold')
ax.set_ylim(88, 100)
ax.legend(fontsize=10, loc='lower right')
ax.grid(axis='y', alpha=0.3)

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.15,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

# Add improvement annotations
improvements = [flux_data[i] - ptera_data[i] for i in range(len(categories))]
for i, imp in enumerate(improvements):
    if imp > 0:
        ax.annotate(f'+{imp:.1f}%', xy=(x[i] + width/2, flux_data[i]),
                    xytext=(x[i] + width/2 + 0.15, flux_data[i] + 0.5),
                    fontsize=9, color='#2ca02c', fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color='#2ca02c', lw=1.2))

plt.tight_layout()
path = os.path.join(fig_dir, 'accuracy_summary.png')
fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
print(f"Saved: {path}")
plt.close(fig)


# ═══════════════════════════════════════════════════════════════
# Figure 3: Feature comparison matrix
# ═══════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(12, 5), facecolor='white')
ax.axis('off')

features = [
    ['Feature', 'PteraSoftware v5.0', 'FLUXVortex (Ours)'],
    ['Compute Backend', 'Numba @njit (CPU)', 'NVIDIA Warp @wp.kernel (GPU)'],
    ['Wing Solver', 'Ring Vortex UVLM', 'Same (reused)'],
    ['Wake Model', 'Ring Vortex Panels', 'Panels + VPM Particles (Hybrid)'],
    ['Far-Field Wake', 'Prescribed / Free panels', 'Free VPM particles (self-induction)'],
    ['Time Integration', 'Euler explicit', 'RK3 low-storage (3rd order)'],
    ['Vortex Stretching', 'None', 'Reformulated VPM (rVPM)'],
    ['Wake Rollup', 'Limited (panel rigidity)', 'Yes (particle dynamics)'],
    ['GPU Acceleration', 'No', 'Yes (Warp kernels)'],
    ['CL Accuracy (Static)', '93.2% vs XFLR5', '96.5% vs XFLR5 (N=10)'],
    ['CL Accuracy (Plunging)', '92-93% vs Theodorsen', '92.5-97.4% vs Theodorsen (N=10)'],
]

cell_colors = []
for i, row in enumerate(features):
    if i == 0:
        cell_colors.append(['#2c5f8a', '#2c5f8a', '#2c5f8a'])
    elif i % 2 == 0:
        cell_colors.append(['#f0f0f0', '#f0f0f0', '#f0f0f0'])
    else:
        cell_colors.append(['white', 'white', 'white'])

table = ax.table(
    cellText=features,
    cellColours=cell_colors,
    cellLoc='center',
    loc='center',
    colWidths=[0.25, 0.35, 0.40],
)

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.0, 1.8)

for i in range(3):
    table[0, i].set_text_props(color='white', fontweight='bold', fontsize=11)

for i in range(1, len(features)):
    table[i, 0].set_text_props(fontweight='bold', fontsize=9, color='#333')

ax.set_title('FLUXVortex vs PteraSoftware: Feature & Accuracy Comparison',
             fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
path = os.path.join(fig_dir, 'feature_comparison.png')
fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
print(f"Saved: {path}")
plt.close(fig)


# ═══════════════════════════════════════════════════════════════
# Figure 4: Architecture diagram (text-based)
# ═══════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(12, 4), facecolor='white')
ax.axis('off')
ax.set_xlim(0, 12)
ax.set_ylim(0, 4)

# Boxes
boxes = [
    (0.3, 2.5, 3.4, 1.2, '#4a90d9', 'PteraSoftware UVLM\n(Bound Vortex Panels)', 10),
    (4.3, 2.5, 3.4, 1.2, '#e8734a', 'Near-Field Wake\n(Ring Panels, N_keep rows)', 10),
    (8.3, 2.5, 3.4, 1.2, '#2ca02c', 'Far-Field Wake\n(VPM Particles, Free)', 10),
    (0.3, 0.5, 3.4, 1.2, '#888888', 'GPU Biot-Savart\n(NVIDIA Warp)', 10),
    (4.3, 0.5, 3.4, 1.2, '#984ea3', 'CL = 92-97% Theodorsen\n(+ Free Wake Rollup)', 10),
    (8.3, 0.5, 3.4, 1.2, '#d62728', 'RK3 + rVPM Stretching\n+ Pedrizzetti Relaxation', 10),
]

for x0, y0, w, h, color, text, fs in boxes:
    rect = FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0.1",
                          facecolor=color, edgecolor='#333', linewidth=1.5, alpha=0.85)
    ax.add_patch(rect)
    ax.text(x0 + w/2, y0 + h/2, text, ha='center', va='center',
            fontsize=fs, color='white', fontweight='bold')

# Arrows
arrow_props = dict(arrowstyle='->', lw=2, color='#333')
ax.annotate('', xy=(4.3, 3.1), xytext=(3.7, 3.1), arrowprops=arrow_props)
ax.annotate('', xy=(8.3, 3.1), xytext=(7.7, 3.1), arrowprops=arrow_props)
ax.annotate('', xy=(2.0, 1.7), xytext=(2.0, 2.5), arrowprops=arrow_props)
ax.annotate('', xy=(6.0, 1.7), xytext=(6.0, 2.5), arrowprops=arrow_props)
ax.annotate('', xy=(10.0, 1.7), xytext=(10.0, 2.5), arrowprops=arrow_props)

# Labels
ax.text(4.0, 3.35, 'N_keep\nrows', ha='center', va='bottom', fontsize=8, color='#555')
ax.text(8.0, 3.35, 'convert\nto particles', ha='center', va='bottom', fontsize=8, color='#555')

ax.set_title('FLUXVortex Hybrid Panel-Particle Architecture',
             fontsize=14, fontweight='bold', pad=10)

plt.tight_layout()
path = os.path.join(fig_dir, 'architecture.png')
fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
print(f"Saved: {path}")
plt.close(fig)

print("\nAll figures generated!")

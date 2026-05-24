"""
Generate flapping wing comparison figure from benchmark results.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

fig_dir = os.path.join(os.path.dirname(__file__), '..', 'figures')

data = np.load(os.path.join(fig_dir, 'flapping_results.npz'), allow_pickle=True)

t_ps = data['t_ps']; cl_ps = data['cl_ps']; cdi_ps = data['cdi_ps']
t_h10 = data['t_h10']; cl_h10 = data['cl_h10']; cdi_h10 = data['cdi_h10']

# Check if FREE variant exists
has_free = len(data['t_h10f']) > 0
if has_free:
    t_h10f = data['t_h10f']; cl_h10f = data['cl_h10f']; cdi_h10f = data['cdi_h10f']

fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor='white')

# ── Panel (a): CL time history ──
ax = axes[0, 0]
ax.plot(t_ps, cl_ps, 'b-', linewidth=1.2, label='PteraSoftware (ring-wake)', alpha=0.9)
ax.plot(t_h10, cl_h10, '--', color='#2ca02c', linewidth=1.0, label='FLUXVortex N=10', alpha=0.8)
if has_free:
    ax.plot(t_h10f, cl_h10f, '-.', color='#e8734a', linewidth=1.0, label='FLUXVortex N=10 FREE', alpha=0.8)
ax.set_xlabel('Time (s)', fontsize=10)
ax.set_ylabel('$C_L$', fontsize=12)
ax.set_title('(a) Lift Coefficient History', fontsize=11, fontweight='bold')
ax.legend(fontsize=8, loc='upper right')
ax.grid(True, alpha=0.3)

# ── Panel (b): CL last cycle zoom ──
ax = axes[0, 1]
period = 1.0
t_start = t_ps[-1] - period
mask_ps = t_ps >= t_start
ax.plot(t_ps[mask_ps], cl_ps[mask_ps], 'b-', linewidth=2.0, label='PteraSoftware', alpha=0.9)

mask_h10 = t_h10 >= t_start
n = min(mask_h10.sum(), mask_ps.sum())
ax.plot(t_h10[mask_h10][:n], cl_h10[mask_h10][:n], 'o-', color='#2ca02c',
        linewidth=1.5, markersize=2, label='FLUXVortex N=10', alpha=0.8)

if has_free:
    mask_h10f = t_h10f >= t_start
    if mask_h10f.any():
        n3 = min(mask_h10f.sum(), mask_ps.sum())
        ax.plot(t_h10f[mask_h10f][:n3], cl_h10f[mask_h10f][:n3], '^-', color='#e8734a',
                linewidth=1.5, markersize=2, label='FLUXVortex N=10 FREE', alpha=0.8)

ax.set_xlabel('Time (s)', fontsize=10)
ax.set_ylabel('$C_L$', fontsize=12)
ax.set_title('(b) Last Cycle Detail', fontsize=11, fontweight='bold')
ax.legend(fontsize=8, loc='upper right')
ax.grid(True, alpha=0.3)

# ── Panel (c): CDi time history ──
ax = axes[1, 0]
ax.plot(t_ps, cdi_ps, 'b-', linewidth=1.2, label='PteraSoftware', alpha=0.9)
ax.plot(t_h10, cdi_h10, '--', color='#2ca02c', linewidth=1.0, label='FLUXVortex N=10', alpha=0.8)
if has_free:
    ax.plot(t_h10f, cdi_h10f, '-.', color='#e8734a', linewidth=1.0, label='FLUXVortex N=10 FREE', alpha=0.8)
ax.set_xlabel('Time (s)', fontsize=10)
ax.set_ylabel('$C_{Di}$', fontsize=12)
ax.set_title('(c) Induced Drag Coefficient History', fontsize=11, fontweight='bold')
ax.legend(fontsize=8, loc='upper right')
ax.grid(True, alpha=0.3)

# ── Panel (d): CL error relative to PteraSoftware ──
ax = axes[1, 1]
n_min = min(len(cl_ps), len(cl_h10))
err_h10 = cl_h10[:n_min] - cl_ps[:n_min]
ax.plot(t_ps[:n_min], err_h10, '-', color='#2ca02c', linewidth=0.8, label='N=10 error', alpha=0.7)
if has_free:
    n_min_f = min(len(cl_ps), len(cl_h10f))
    err_h10f = cl_h10f[:n_min_f] - cl_ps[:n_min_f]
    ax.plot(t_ps[:n_min_f], err_h10f, '-', color='#e8734a', linewidth=0.8, label='N=10 FREE error', alpha=0.7)
ax.axhline(0, color='k', linewidth=0.5)
ax.set_xlabel('Time (s)', fontsize=10)
ax.set_ylabel('$\\Delta C_L$ (vs PteraSoftware)', fontsize=10)
ax.set_title('(d) CL Difference from PteraSoftware', fontsize=11, fontweight='bold')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Compute and display correlation
corr_h10 = np.corrcoef(cl_ps[:n_min], cl_h10[:n_min])[0, 1]
rmse_h10 = np.sqrt(np.mean((cl_ps[:n_min] - cl_h10[:n_min])**2))

stats_text = f'N=10: corr={corr_h10:.6f}, RMSE={rmse_h10:.4f}'
if has_free:
    n_min_f = min(len(cl_ps), len(cl_h10f))
    corr_h10f = np.corrcoef(cl_ps[:n_min_f], cl_h10f[:n_min_f])[0, 1]
    rmse_h10f = np.sqrt(np.mean((cl_ps[:n_min_f] - cl_h10f[:n_min_f])**2))
    stats_text += f'   |   N=10 FREE: corr={corr_h10f:.6f}, RMSE={rmse_h10f:.4f}'

fig.text(0.5, 0.01, stats_text, ha='center', fontsize=10, style='italic', color='#555')

fig.suptitle('FLUXVortex vs PteraSoftware: Flapping Wing (15° sweep, V=10 m/s, 3 cycles)',
             fontsize=13, fontweight='bold', y=0.99)

plt.tight_layout(rect=[0, 0.03, 1, 0.97])
path = os.path.join(fig_dir, 'flapping_comparison.png')
fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
print(f"Saved: {path}")
plt.close(fig)

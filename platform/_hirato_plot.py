"""Plot the Hirato Case-1 reproduction (C_L vs t*) from hirato_case1.npz, styled like paper Fig.15a:
LEV-ON (faithful vortex-sheet) vs LEV-OFF (attached), with alpha(t*) overlaid. Marks the LEV-onset and the
lift-growth-rate reduction that the faithful LEV produces."""
import os, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

d = np.load(os.path.join('docs', 'hirato_case1.npz'))
ts = d['tstar']; alpha = d['alpha']; off = d['off_b']; on = d['on_b']

def smooth(a, w=5):
    k = np.ones(w) / w; return np.convolve(np.pad(a, w // 2, mode='edge'), k, mode='valid')[:len(a)]

fig, ax = plt.subplots(figsize=(8, 5.2)); ax2 = ax.twinx()
ax2.plot(ts, alpha, color='0.6', lw=1.2, ls='-', label='α (deg)')
ax2.set_ylabel('angle of attack α (deg)', color='0.5'); ax2.set_ylim(-5, 50)
ax.plot(ts, smooth(off), 'k--', lw=1.8, label='UVLM without LEV (attached)')
ax.plot(ts, smooth(on), 'b-', lw=2.0, label='UVLM with LEV (faithful Hirato vortex-sheet)')
# LEV onset ~ where the two diverge
div = np.where(np.abs(on - off) > 0.15)[0]
if len(div):
    t_on = ts[div[0]]; ax.axvline(t_on, color='r', ls=':', lw=1.2)
    ax.text(t_on + 0.03, ax.get_ylim()[1] * 0.1 if False else 0.3, f'  LEV onset t*≈{t_on:.2f}', color='r', fontsize=9)
ax.axhline(0, color='0.8', lw=0.6)
ax.set_xlabel('nondimensional time  t* = U t / c'); ax.set_ylabel('wing lift coefficient  C_L')
ax.set_title('Hirato Case 1 reproduction: SD7003 rect AR=6, pitch ramp 0→45° K=0.3, Re=20k, LESP_crit=0.27\n'
             '(cf. paper Fig.15a — faithful LEV reduces the lift-growth rate vs attached UVLM)', fontsize=9.5)
ax.legend(loc='upper left', fontsize=8.5); ax.grid(alpha=0.3)
ax.set_xlim(0, ts[-1])
pk_off = off.max(); pk_on = on.max()
ax.text(0.02, 0.02, f'peak C_L: attached={pk_off:.2f}, with-LEV={pk_on:.2f}  (paper peak ~4)',
        transform=ax.transAxes, fontsize=8, va='bottom')
fig.tight_layout(); out = os.path.join('docs', 'hirato_case1_CL.png'); fig.savefig(out, dpi=120)
print(f'saved {out}   peak CL off={pk_off:.2f} on={pk_on:.2f}', flush=True)

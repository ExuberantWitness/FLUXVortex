"""Decisive: does the matrix column match the particle field for the same ring?"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/tmp/fluxv-v5-nextgen/src")
sys.path.insert(0, "/tmp/fluxv-v5-nextgen/platform")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pfield import ParticleField
from bing_joint_solver import ring_velocity

# a ring in pterasoftware vertex order (Fr, Fl, Bl, Br), like the LEV ring
# station s at y=0, s+1 at y=0.6; LE line at x=0, offset legs +0.1*normal(+z)
rv = np.array([
    [0.0, 0.6, 0.0],   # Fr on LE (hi station)
    [0.0, 0.0, 0.0],   # Fl on LE (lo station)
    [0.0, 0.0, 0.1],   # Bl offset
    [0.0, 0.6, 0.1],   # Br offset
])
rings = rv[None, :, :]

targets = np.array([[0.3, 0.3, 0.0], [0.6, 0.3, 0.0]])  # collocation-ish points
norms = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])

# matrix column (my ring_velocity on their-ordered array)
col = np.einsum("ij,ikj->ik", norms, ring_velocity(targets, rings))[ :, 0]

# particle field, unit strength, reverse traversal (leg -> leg-1)
pf = ParticleField(capacity=100)
pos, gam, sig = [], [], []
for leg in range(4):
    o = rv[leg]
    t = rv[(leg - 1) % 4]
    vec = t - o
    sig.append(np.linalg.norm(vec) / 17.5)
    pos.append(0.5 * (o + t))
    gam.append(vec * 1.0)
pf.add_particles(np.array(pos), np.array(gam), np.array(sig))
v = pf.velocity_at(targets)
vn = np.einsum("ij,ij->i", v, norms)

print("column (ring_velocity . n):", np.round(col, 5))
print("particle field  (v . n)   :", np.round(vn, 5))
print("ratio:", np.round(vn / col, 3))

# forward traversal for comparison
pf2 = ParticleField(capacity=100)
pos, gam, sig = [], [], []
for leg in range(4):
    o = rv[leg]
    t = rv[(leg + 1) % 4]
    vec = t - o
    sig.append(np.linalg.norm(vec) / 17.5)
    pos.append(0.5 * (o + t))
    gam.append(vec * 1.0)
pf2.add_particles(np.array(pos), np.array(gam), np.array(sig))
v2 = pf2.velocity_at(targets)
print("forward-traversal particles:", np.round(np.einsum("ij,ij->i", v2, norms), 5))

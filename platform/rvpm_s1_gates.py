"""rVPM S1 验证门 v2:
G1a 核约定无关门:U vs ln(1/σ) 斜率 = Γ/(4πR)(解析,约定无关);
G1b Saffman 值(a=√2σ,高斯涡量核约定);
G2 leapfrog 判别版:relax=0、t=40、n=150/环、R 振荡与 |α| 增长诊断,cVPM vs rVPM。"""
import sys, os, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, "..", "src"), HERE]
from fluxvortex.warp_vpm_rvpm import make_ring, step_lsrk3

def ring_speed(a, n=200, nsteps=150, dt=0.01):
    X, G, sig = make_ring(R=1.0, Gamma=1.0, a=a, n=n)
    z0 = X[:, 2].mean()
    for _ in range(nsteps):
        X, G, sig = step_lsrk3(X, G, sig, dt, relax=0.0)
    return (X[:, 2].mean() - z0) / (nsteps * dt)

print("== G1 涡环速度(约定无关斜率门 + Saffman 值门)==", flush=True)
sigs = [0.05, 0.08, 0.12, 0.2]
Us = [ring_speed(a) for a in sigs]
sl = np.polyfit(np.log(1.0 / np.array(sigs)), Us, 1)[0]
print(f"  U vs ln(1/σ) 斜率 = {sl:.4f} vs 解析 Γ/(4πR) = {1/(4*np.pi):.4f} "
      f"(偏差 {(sl/(1/(4*np.pi))-1)*100:+.1f}%)", flush=True)
for a, U in zip(sigs, Us):
    Ua = 1/(4*np.pi)*(np.log(8/(np.sqrt(2)*a)) - 0.558)
    print(f"  σ={a}: U={U:.4f} vs Saffman(a=√2σ) {Ua:.4f} ({(U/Ua-1)*100:+.1f}%)", flush=True)

print("== G2 leapfrog 判别版(relax=0, t=40, dt=0.02, n=150/环)==", flush=True)
def leapfrog(f, g, nsteps=2000, dt=0.02, n=150):
    X1, G1, s1 = make_ring(R=1.0, z=0.0, n=n)
    X2, G2, s2 = make_ring(R=1.0, z=0.6, n=n)
    X = np.vstack([X1, X2]); G = np.vstack([G1, G2]); sig = np.concatenate([s1, s2])
    a0 = np.linalg.norm(G, axis=1).mean()
    Rmin, Rmax, amax_t = 9e9, 0.0, 0.0
    t0 = time.time()
    for i in range(nsteps):
        X, G, sig = step_lsrk3(X, G, sig, dt, f=f, g=g, relax=0.0)
        if not np.isfinite(X).all() or np.abs(X).max() > 200:
            return f"发散 @t={i*dt:.1f}", time.time()-t0
        if i % 50 == 0:
            r1 = np.hypot(X[:n,0], X[:n,1]).mean(); r2 = np.hypot(X[n:,0], X[n:,1]).mean()
            Rmin = min(Rmin, r1, r2); Rmax = max(Rmax, r1, r2)
            amax_t = max(amax_t, np.linalg.norm(G, axis=1).max()/a0)
    return (f"存活到 t={nsteps*dt:.0f};R 振荡 [{Rmin:.2f},{Rmax:.2f}] "
            f"|α|max/初值 {amax_t:.2f} σ均 {sig.mean():.3f}"), time.time()-t0
for f, g, tag in ((0.0, 0.2, "rVPM"), (0.0, 0.0, "cVPM 消融")):
    msg, tt = leapfrog(f, g)
    print(f"  [{tag}] {msg}  ({tt:.0f}s)", flush=True)

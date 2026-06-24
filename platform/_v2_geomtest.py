import numpy as np, sys
sys.path.insert(0,'.'); sys.path.insert(0,'..')
import flap_flight_validate as ffv
import robowing as rw
U=8.0; chord=0.287; hs=0.80
geoms = {
  'rectangular uniform': lambda nc,ns,c,h: ffv.flat_wing(nc,ns,c,h),
  'rect + cosine chord':  lambda nc,ns,c,h: rw.robowing(nc,ns,c,h,round_tip=False,cosine_chord=True),
  'ROUNDED + cosine':     lambda nc,ns,c,h: rw.robowing(nc,ns,c,h,round_tip=True,cosine_chord=True),
}
# steady 5deg + flapping 2Hz, both wings
for name, gf in geoms.items():
    ffv.flat_wing = gf  # monkeypatch geometry into gpu_run
    area = (rw.area_both(chord,hs,round_tip=('ROUND' in name)) if 'rect' not in name.split()[0] or 'ROUND' in name else 2*chord*hs)
    rs = ffv.gpu_run(nc=8,ns=16,chord=chord,half_span=hs,mass=0.7,U=U,aoa_deg=5.0,flap_amp_deg=0.0,freq=1.0,n_cycle=6,steps_per_cycle=40,verbose=False)
    rf = ffv.gpu_run(nc=8,ns=16,chord=chord,half_span=hs,mass=0.7,U=U,aoa_deg=5.0,flap_amp_deg=45.0,freq=2.0,n_cycle=5,steps_per_cycle=40,verbose=False)
    print(f"  {name:22s}: steady L={rs['L']:.2f}N  flapping L={rf['L']:.2f}N T={rf['T']:.2f}N",flush=True)
print("paper @8m/s 5deg 2Hz twist0: lift ~6.6N",flush=True)
print("DONE",flush=True)

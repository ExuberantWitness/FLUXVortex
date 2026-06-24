import sys; sys.path.insert(0,'.'); sys.path.insert(0,'..')
import numpy as np
import flex_aircraft as F
ac = F.FlexAircraft(amp_deg=22.0, flap_hz=3.0, E0=50e9, body_aoa_deg=45.0, V0=6.0, trim_aoa_deg=6.0, substeps=16)
for i in range(20):
    o = ac.step_window()
    d = np.abs(ac.entry.state()['verts'][...,2]).max()
    print(f"w{i:2d} L={o['F_lift']:+12.2f} T={o['F_thrust']:+10.2f} defl={d:.5f}m")

from _v2_flap_strip import flapping_wing
GF=9.81/1000
# Fig 18b @ 8 m/s: data lift (g) vs freq
data = {1.4:626.1, 1.7:689.9, 2.0:759.4, 2.3:794.2, 2.6:788.4}
print("Strip-LDVM (attached, dihedral AoA) vs RoboEagle Fig18b lift @8m/s, 5deg, flap=45, twist0:")
print(f"{'freq':>5} {'data(N)':>8} {'strip(N)':>9} {'ratio':>6}")
for fq in (1.4, 1.7, 2.0, 2.3, 2.6):
    r = flapping_wing(U=8.0, aoa_deg=5.0, flap_amp_deg=45.0, twist_amp_deg=0.0, freq=fq,
                      lesp_crit=99.0, dihedral_project=True, dihedral_aoa=True,
                      ns=6, nc=40, n_cycle=4, steps_per_cycle=80)
    dN = data[fq]*GF
    print(f"{fq:5.1f} {dN:8.2f} {r['L']:9.2f} {r['L']/dN:6.2f}", flush=True)
print("DONE")

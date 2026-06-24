from _v2_flap_strip import flapping_wing
# strip-LDVM at flap=45, 5deg, twist0, 2.3Hz: with vs without the dihedral AoA reduction
# compare to 3D UVLM (1.82N) and steady (~6.8N), paper ~7.8N
for da in (False, True):
    r = flapping_wing(U=8.0, aoa_deg=5.0, flap_amp_deg=45.0, twist_amp_deg=0.0, freq=2.3,
                      lesp_crit=99.0, dihedral_project=True, dihedral_aoa=da)
    tag = "dihedral AoA reduction ON " if da else "dihedral AoA reduction OFF"
    print(f"  strip-LDVM {tag}: L={r['L']:+6.2f}N  T={r['T']:+6.2f}N   (3D UVLM=1.82N, steady~6.8N, paper~7.8N)")
print("DONE")

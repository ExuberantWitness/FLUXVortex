# Final v4b provenance supplement

The individual run manifests hash their principal runner, data inputs, and
result tables. This supplement closes the direct-dependency and publication-
figure gaps identified during the final integrity audit. It is a deterministic
existence/integrity record, not evidence that the scientific claim is correct.

| Repository-relative path | SHA-256 |
|---|---|
| `platform/forward_flight_benchmarks/cases.py` | `56ae0b20c247d2dac822a1ebb9332bdae24367c76bc4fd3f7783729baace33a9` |
| `platform/forward_flight_benchmarks/ptera_adapter.py` | `c1122aab49867a5968bdaeb15556d737be2a22c8eba1252cc71f6f0c6f1f81a9` |
| `platform/flap_ldvm.py` | `107d83576cad0a2b7b16c8de29a6d3c17b1e1493321fdfec549545a9987e91e1` |
| `platform/ldvm_fourier.py` | `bf81c46e10f70c005cdde0a79c180592305334469d3f1c1c45282b265f81030d` |
| `src/fluxvortex/solver.py` | `1fab59f268a8af1c3935ebd2210012c2f13ab452df210268a5b3fb3324ae3b94` |
| `platform/forward_flight_benchmarks/causal_incidence_owner.py` | `1fdfba4c3a270f2cddf18cd43cd7de0c7c691dddd4c7c971faf8e12072c9b901` |
| `platform/forward_flight_benchmarks/ldvm_uvlm_correction.py` | `d72d794242ab702bcd95ab1f5aaf6e623a87722eb1474f073124f6f60031db13` |
| `platform/forward_flight_benchmarks/run_v4_crosspaper.py` | `1b227af5ef03959b39ba8150fa0d066e33be5cda14db1133b32dcd5998590667` |
| `platform/forward_flight_benchmarks/run_stevens2017_benchmark.py` | `0e0aa6f903f697908863667c3a18bbba5d0720dce03b34cfa10269f41e9b300f` |
| `platform/forward_flight_benchmarks/run_fig14_phase_cache.py` | `8b8dc6cb492fc947e7696a4702fd5e2a362f17e9a1eae3d33a40d6f6f997abf6` |
| `platform/forward_flight_benchmarks/stevens2017.py` | `564ec631a9f9810b2ec6b03a2bf33c2441f1a585f3479f826e630e6485613416` |
| `platform/forward_flight_benchmarks/plot_v4b_comparison.py` | `0fe05599b58800b88a1c4f367471b72dace9302535dbf38d837afb5877923f6d` |
| `docs/forward_flight_large_pitch/reproductions/plev2025/source_data/yang2025_fig11_rigid_digitized.csv` | `0351a59d601513a2c0e1605863f3afeeb41bf9d0dc57e73cc54b5b9167d2a8ea` |
| `docs/forward_flight_large_pitch/reproductions/unified_fluxv_upgrade_20260812/source_data/izraelevitz2017_fig11_digitized.csv` | `b3fd20cea68d61664720425b9c7da27d4b5d87a89c80d7296c04887860e141bc` |
| `docs/forward_flight_large_pitch/reproductions/unified_fluxv_upgrade_20260812/source_data/izraelevitz2017_fig14_digitized.csv` | `993f410c5d4857a221e57c616bf45beb5eaef5391a2deafb0b6e48e6d083b3cf` |
| `docs/forward_flight_large_pitch/reproductions/unified_fluxv_v4_ldvm_stevens_20260812/runs/20260812_fluxv_v4b_crosspaper_full/figures/fluxv_v4b_crosspaper_comparison.png` | `2a67846b15f1df418e264e3f44e1490a3baeed4b00d5e71cb6efe491d4a0bd81` |
| `docs/forward_flight_large_pitch/reproductions/unified_fluxv_v4_ldvm_stevens_20260812/runs/20260812_fluxv_v4b_crosspaper_full/figures/fluxv_v4b_crosspaper_comparison.pdf` | `f759bc4186cf15658ef22bc39dc5586750f45e68188cc860c1f1c168ce77df75` |
| `docs/forward_flight_large_pitch/reproductions/unified_fluxv_v4_ldvm_stevens_20260812/runs/20260812_stevens2017_v4b_full/figures/stevens2017_lift_comparison.png` | `a9ba118730604f685da41e94fbcb416981495532163a800c2d29492aa37839eb` |
| `docs/forward_flight_large_pitch/reproductions/unified_fluxv_v4_ldvm_stevens_20260812/runs/20260812_stevens2017_v4b_full/figures/stevens2017_lift_comparison.pdf` | `2784d1d362ecd3ebd28158a770143896af99ddf8bd56cad9828f4e8a46a39a57` |

The Stevens paper PDF is intentionally not vendored in this publication
branch. Its public bibliographic/source URL and digitization method are recorded
in `source_data/DIGITIZATION_STEVENS2017.md`.

The execution environment used Python 3.12.13 with NumPy 2.4.6,
PteraSoftware 5.0.0, FluxVortex 0.9.0, and Numba 0.65.1. This supplement is
reviewed by deterministic re-hashing before publication, but is not self-
authenticating; the Git commit that contains it is the outer immutable record.

# Run Log Summary

- Tests-first run: `5 failed`; source-strip storage and strip impulse outputs
  did not yet exist.
- First implementation run: `4 passed / 1 failed`; the only failure was a
  test tolerance of `3.0e-13` versus an observed grouped-reduction difference
  of `3.197442310920451e-13`. The production scaled closure gate was unchanged;
  the test was replaced by a scaled float64 tolerance and an independent
  per-strip formula oracle.
- Final focused run: `5/5 PASS`, zero warnings.
- Particle/active-LEV/incremental integration run: `26/26 PASS`.
- Full selected Q16/LEV/TEV/FSI surface: `146/146 PASS` in `19.28 s`.
- Black, Ruff, py_compile and diff/whitespace checks passed.
- The experiment skill's managed bash/artifact/memory tools were unavailable;
  local non-interactive commands and this repository package are the disclosed
  fallback.
- No paper data, GT or scorer was accessed.

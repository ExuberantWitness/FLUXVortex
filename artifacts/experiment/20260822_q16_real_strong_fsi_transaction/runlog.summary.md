# Run Log Summary

- Initial coupling runs exposed two real integration defects: missing zero-LEV
  strip records and a high-E reference residual floor.
- The corrected nonzero 5-degree pilot converged without relaxing the frozen
  outer tolerance; the structural budget was raised to 15 Newton / 1024 CG
  iterations because the current operator remains unpreconditioned.
- Focused baseline passed `4/4 in 152.58 s`; injected second-evaluation failure
  passed `1/1 in 7.49 s`; selected joint regression passed `157/157`.
- Touched files passed Black, Ruff and py_compile.
- No full matrix, paper data, GT or scorer was accessed.

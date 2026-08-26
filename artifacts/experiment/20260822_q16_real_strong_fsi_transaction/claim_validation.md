# Claim Validation

| Claim | Evidence | Verdict |
|---|---|---|
| complete real aero load reaches Q16 Newton solve | resolved+LEV transfer and real coupling test | supported |
| separated LEV, joint TEV and free wake remain active | committed solver state/counters and mandatory-mode gates | supported |
| one accepted step advances both owners once | generation/pointer/wake assertions | supported |
| failed/nonconverged step advances neither owner | injected failure, forced nonconvergence and clean retry | supported |
| multi-step FSI is stable/accurate | not run | unsupported |
| classic FSI experiment is reproduced | not run | unsupported |

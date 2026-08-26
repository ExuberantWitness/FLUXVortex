# Q16 CUDA Newmark--Newton--CG checklist

- [x] Method, GPU boundary and stop conditions frozen.
- [x] RED: structural stepper absent.
- [x] GREEN: zero-load stationary batch passes.
- [x] GREEN: loaded residual/boundary/output gates pass.
- [x] GREEN: forced failure leaves inputs unchanged and clean retry passes.
- [x] Host/dtype/shape/nonfinite/delta-time attacks fail closed.
- [x] Joint/static checks and exact hashes recorded.

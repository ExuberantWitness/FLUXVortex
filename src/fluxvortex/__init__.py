# FLUXVortex — GPU-Accelerated Hybrid Panel-Particle Vortex Method Solver
#
# GPU acceleration available via NVIDIA Warp:
#   from fluxvortex.warp_patch import patch, unpatch
#   patch()    # activate GPU
#   unpatch()  # restore CPU
#
# Aeroelastic simulation:
#   from fluxvortex.aeroelastic_solver import AeroelasticSolver

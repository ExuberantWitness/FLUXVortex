# FLUXVortex — GPU-Accelerated Hybrid Panel-Particle Vortex Method Solver
#
# GPU acceleration available via NVIDIA Warp:
#   from fluxvortex.warp_patch import patch, unpatch
#   patch()    # activate GPU
#   unpatch()  # restore CPU
#
# Aeroelastic simulation (beam FE):
#   from fluxvortex.aeroelastic_solver import AeroelasticSolver
#
# Aeroelastic simulation (XPBD particle-mesh, implicit coupling):
#   from fluxvortex.aero_coupling import ParticleMeshAeroelasticSolver
#
# Unified 2D particle-mesh structural dynamics (XPBD):
#   from fluxvortex.particle_mesh import ParticleMesh, create_wing_mesh

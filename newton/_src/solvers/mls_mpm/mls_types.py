import warp as wp

@wp.struct
class MLSMPMParameters:
    support_radius: float
    viscosity: float
    density: float
    bulk_modulus: float
    gravity: wp.vec3
    grid_origin: wp.vec3
    grid_dx: float
    grid_dim: wp.vec3i
    collider_active: int  # 0: 无, 1: 有
    collider_center: wp.vec3
    collider_half_size: wp.vec3
    collider_friction: float

@wp.struct
class MLSMPMParticleData:
    position: wp.array(dtype=wp.vec3)
    velocity: wp.array(dtype=wp.vec3)
    C: wp.array(dtype=wp.mat33)
    mass: wp.array(dtype=float)
    volume: wp.array(dtype=float)
    deformation_gradient: wp.array(dtype=wp.mat33)
    stress: wp.array(dtype=wp.mat33)
    J: wp.array(dtype=float)

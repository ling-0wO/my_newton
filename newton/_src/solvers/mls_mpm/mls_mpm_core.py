import warp as wp
import numpy as np
@wp.struct
class MLSMPMParameters:
    support_radius: float
    viscosity: float
    surface_tension: float  # 表面张力系数
    density: float
    gravity: wp.vec3
    kernel_type: int

@wp.struct
class MLSMPMParticleData:
    position: wp.array(dtype=wp.vec3)
    velocity: wp.array(dtype=wp.vec3)
    mass: wp.array(dtype=float)
    volume: wp.array(dtype=float)
    deformation_gradient: wp.array(dtype=wp.mat33)
    stress: wp.array(dtype=wp.mat33)

# 核函数实现
@wp.func
def cubic_kernel(r: float, h: float):
    q = r / h
    if q <= 1.0:
        return 1.0 - 1.5 * q * q + 0.75 * q * q * q
    elif q <= 2.0:
        t = 2.0 - q
        return 0.25 * t * t * t
    else:
        return 0.0

@wp.func
def cubic_kernel_gradient(r_vec: wp.vec3, h: float):
    r = wp.length(r_vec)
    if r < 1e-10:
        return wp.vec3(0.0)

    q = r / h
    if q <= 1.0:
        factor = (-3.0 + 2.25 * q) / (h * h)
    elif q <= 2.0:
        t = 2.0 - q
        factor = -0.75 * t * t / (h * h * r)
    else:
        factor = 0.0

    return factor * r_vec

# 移动最小二乘权重函数
@wp.func
def mls_weight(r_vec: wp.vec3, h: float, node_pos: wp.vec3, particle_pos: wp.vec3):
    # 基础核函数值
    kernel_val = cubic_kernel(wp.length(r_vec), h)

    # MLS修正：构建形函数矩阵
    dx = node_pos[0] - particle_pos[0]
    dy = node_pos[1] - particle_pos[1]
    dz = node_pos[2] - particle_pos[2]

    p = wp.vec4(1.0, dx / h, dy / h, dz / h)
    weight = kernel_val * (1.0 + 0.1 * (dx * dx + dy * dy + dz * dz) / (h * h))
    return wp.max(weight, 0.0)
class MLSMPMCore:
    def __init__(self, device="cuda:0"):
        self.device = device
        self.params = MLSMPMParameters()
        self.params.support_radius = 0.1
        self.params.viscosity = 0.001
        self.params.surface_tension = 0.072
        self.params.density = 1000.0
        self.params.gravity = wp.vec3(0.0, 0.0, -9.8)
        self.params.kernel_type = 0

    # 初始化粒子
    def initialize_particles(self, positions, velocities, mass):
        num_particles = len(positions)

        particle_data = MLSMPMParticleData()
        particle_data.position = wp.array(positions, dtype=wp.vec3, device=self.device)
        particle_data.velocity = wp.array(velocities, dtype=wp.vec3, device=self.device)
        particle_data.mass = wp.array([mass] * num_particles, dtype=float, device=self.device)

        # 体积
        volumes = np.ones(num_particles) * (mass / self.params.density)
        particle_data.volume = wp.array(volumes, dtype=float, device=self.device)

        # 变形梯度和应力
        identity = wp.mat33(np.eye(3))
        particle_data.deformation_gradient = wp.full(num_particles, identity, dtype=wp.mat33, device=self.device)
        particle_data.stress = wp.zeros(num_particles, dtype=wp.mat33, device=self.device)

        return particle_data

    # 计算MLS权重核函数
    @wp.kernel
    def compute_mls_weights_kernel(
            particle_pos: wp.array(dtype=wp.vec3),
            node_pos: wp.array(dtype=wp.vec3),
            support_radius: float,
            weights: wp.array(dtype=float),
            weight_gradients: wp.array(dtype=wp.vec3)
    ):
        node_id, particle_id = wp.tid()

        r_vec = node_pos[node_id] - particle_pos[particle_id]
        r = wp.length(r_vec)

        if r < support_radius:
            kernel_val = cubic_kernel(r, support_radius)
            kernel_grad = cubic_kernel_gradient(r_vec, support_radius)

            weight = mls_weight(r_vec, support_radius, node_pos[node_id], particle_pos[particle_id])
            weight_grad = kernel_grad

            weights[node_id * wp.tid().y + particle_id] = weight
            weight_gradients[node_id * wp.tid().y + particle_id] = weight_grad

    # 计算MLS权重矩阵
    def compute_mls_weights(self, particle_pos, node_pos):
        num_nodes = len(node_pos)
        num_particles = len(particle_pos)

        weights = wp.zeros(num_nodes * num_particles, dtype=float, device=self.device)
        weight_gradients = wp.zeros(num_nodes * num_particles, dtype=wp.vec3, device=self.device)

        wp.launch(
            self.compute_mls_weights_kernel,
            dim=[num_nodes, num_particles],
            inputs=[particle_pos, node_pos, self.params.support_radius, weights, weight_gradients],
            device=self.device
        )

        return weights, weight_gradients


# 流体本构模型(替换原弹性模型）
@wp.func
def fluid_constitutive_model(
        deformation_grad: wp.mat33,
        velocity_grad: wp.mat33,
        density: float,
        viscosity: float
) -> wp.mat33:

    # 计算体积变化率
    J = wp.determinant(deformation_grad)
    volume_ratio = wp.max(J, 0.01)  # 防止除零

    # 计算压力 (理想气体状态方程简化)
    pressure = 1000.0 * (wp.pow(volume_ratio, 7.0) - 1.0)

    # 计算应变率张量
    strain_rate = 0.5 * (velocity_grad + wp.transpose(velocity_grad))

    # 牛顿流体应力：σ = -pI + 2μD
    stress = -pressure * wp.identity(n=3, dtype=float) + 2.0 * viscosity * strain_rate

    return stress


@wp.kernel
def update_particle_stress_kernel(
        particle_data: MLSMPMParticleData,
        density: float,
        viscosity: float
):
    particle_id = wp.tid()

    # TODO：假设速度梯度已知,实际应网格插值得到
    vel_grad = wp.mat33(0.0)

    particle_data.stress[particle_id] = fluid_constitutive_model(
        particle_data.deformation_gradient[particle_id],
        vel_grad,
        density,
        viscosity
    )
@fem.integrand
def integrate_velocity_mls(
    s: fem.Sample,
    domain: fem.Domain,
    u: fem.Field,
    mls_core: MLSMPMCore,
    velocity_gradients: wp.array(dtype=wp.mat33),
    inv_cell_volume: float,
    particle_density: wp.array(dtype=float),
    particle_flags: wp.array(dtype=wp.int32),
):
    # 使用MLS权重而不是标准FEM形函数
    weight = mls_core.mls_weight_kernel(...)
    # MLS-MPM特定的动量映射
    return mls_momentum_mapping(...)


class SolverMLSMPM(SolverImplicitMPM):
    """基于MLS-MPM的流体求解器"""

    def __init__(self, model, options):
        super().__init__(model, options)
        self.mls_core = MLSMPMCore(self._extract_mls_parameters(options))

    def _step_impl(self, state_in, state_out, dt, scratch):
        # 重写步进函数，使用MLS-MPM算法
        # 1. MLS粒子到网格映射
        # 2. MLS网格求解
        # 3. MLS网格到粒子映射
        pass
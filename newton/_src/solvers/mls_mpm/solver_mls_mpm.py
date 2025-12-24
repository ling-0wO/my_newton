import warp as wp
import numpy as np
from ..implicit_mpm import SolverImplicitMPM

__all__ = ["SolverMLSMPM"]
CUBE_FULL_EXTENTS = (0.5, 0.5, 0.5)
CUBE_CENTER = (0.25, 0.25, 0.25)

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
    collider_active: int
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


class MLSMPMCore:
    def __init__(self, device="cuda:0"):
        self.device = device
        self.params = MLSMPMParameters()
        self.params.support_radius = 0.15
        self.params.viscosity = 0.01
        self.params.density = 1000.0
        self.params.gravity = wp.vec3(0.0, 0.0, -9.8)
        self.params.grid_dx = 0.15
        self.params.grid_origin = wp.vec3(-10.0, -10.0, -2.0)
        self.params.grid_dim = wp.vec3i(128, 128, 128)

    def initialize_particles(self, positions, velocities, mass):
        num_particles = len(positions)
        particle_data = MLSMPMParticleData()
        particle_data.position = wp.array(positions, dtype=wp.vec3, device=self.device)
        particle_data.velocity = wp.array(velocities, dtype=wp.vec3, device=self.device)

        if isinstance(mass, (float, int)):
            mass_arr = wp.full(num_particles, float(mass), dtype=float, device=self.device)
        else:
            mass_arr = wp.array(mass, dtype=float, device=self.device)
        particle_data.mass = mass_arr

        rho = self.params.density
        if isinstance(mass, (float, int)):
            vol_val = float(mass) / rho
            particle_data.volume = wp.full(num_particles, vol_val, dtype=float, device=self.device)
        else:
            vol_np = np.array(mass) / rho
            particle_data.volume = wp.array(vol_np, dtype=float, device=self.device)

        identity = wp.mat33(np.eye(3))
        particle_data.deformation_gradient = wp.full(num_particles, identity, dtype=wp.mat33, device=self.device)
        particle_data.C = wp.zeros(num_particles, dtype=wp.mat33, device=self.device)
        particle_data.stress = wp.zeros(num_particles, dtype=wp.mat33, device=self.device)
        particle_data.J = wp.ones(num_particles, dtype=float, device=self.device)

        return particle_data

@wp.func
def sd_box(p: wp.vec3, b: wp.vec3):
    d = wp.vec3(wp.abs(p[0]) - b[0], wp.abs(p[1]) - b[1], wp.abs(p[2]) - b[2])
    # 外部距离 + 内部距离
    outside = wp.length(wp.max(d, wp.vec3(0.0)))
    inside = wp.min(wp.max(d[0], wp.max(d[1], d[2])), 0.0)
    return outside + inside


@wp.func
def get_box_normal(p: wp.vec3, b: wp.vec3):
    bias = 1.0e-5

    nx = 0.0
    ny = 0.0
    nz = 0.0

    dx = wp.abs(p[0]) - b[0]
    dy = wp.abs(p[1]) - b[1]
    dz = wp.abs(p[2]) - b[2]

    if dx > dy and dx > dz:
        nx = wp.sign(p[0])
    elif dy > dz:
        ny = wp.sign(p[1])
    else:
        nz = wp.sign(p[2])

    return wp.vec3(nx, ny, nz)

@wp.func
def fluid_constitutive_model(J: float, C: wp.mat33, deformation_grad: wp.mat33, viscosity: float,
                             bulk_modulus: float) -> wp.mat33:
    gamma = 7.0
    # 限制 J 的范围，防止 NaN，但允许一定的压缩
    J_clamped = wp.clamp(J, 0.1, 1.5)

    # Murnaghan-Tait EOS
    pressure = bulk_modulus * (wp.pow(J_clamped, -gamma) - 1.0)

    # === 关键修复：消除负压 (Tensile Instability) ===
    # 水不能承受拉力。如果不加这个，自由表面的粒子会因为负压粘在一起，
    # 积聚能量后突然炸开。
    if pressure < 0.0:
        pressure = 0.0

    # 粘性应力
    strain_rate = 0.5 * (C + wp.transpose(C))
    viscous_stress = 2.0 * viscosity * strain_rate

    stress = -pressure * wp.identity(n=3, dtype=float) + viscous_stress
    return stress


@wp.kernel
def update_particle_stress_kernel(particle_data: MLSMPMParticleData, viscosity: float, bulk_modulus: float):
    p = wp.tid()
    particle_data.stress[p] = fluid_constitutive_model(
        particle_data.J[p],
        particle_data.C[p],
        particle_data.deformation_gradient[p],
        viscosity,
        bulk_modulus
    )


class MLSMPMOptions(SolverImplicitMPM.Options):
    def __init__(self):
        super().__init__()
        self.support_radius = 0.15
        self.viscosity = 0.05  # 稍微增加粘度以稳定
        self.fluid_density = 1000.0
        self.bulk_modulus = 1.0e6  # 1M Pa 足够演示，配合自适应步长


class SolverMLSMPM(SolverImplicitMPM):
    Options = MLSMPMOptions

    def __init__(self, model, options: MLSMPMOptions):
        super().__init__(model, options)
        self.mls_core = MLSMPMCore(device=model.device)
        self.mls_core.params.viscosity = options.viscosity
        self.mls_core.params.density = options.fluid_density
        self.mls_core.params.support_radius = options.support_radius
        self.mls_core.params.grid_dx = options.support_radius
        self.mls_core.params.bulk_modulus = options.bulk_modulus

        self._initialized = False
        self.mls_particle_data = None
        self.grid_mass = None
        self.grid_mom = None

    def initialize_mls_particles(self, state):
        try:
            positions = state.particle_q.numpy() if hasattr(state, 'particle_q') else []
            velocities = state.particle_qd.numpy() if hasattr(state, 'particle_qd') else []

            if len(positions) == 0: return False

            if hasattr(self.model, 'particle_mass'):
                masses = self.model.particle_mass.numpy()
            else:
                masses = np.ones(len(positions), dtype=np.float32)

            # Padding
            padding = 0.5
            min_pos = np.min(positions, axis=0) - padding
            max_pos = np.max(positions, axis=0) + padding
            # 稍微降低地板边界，防止初始粒子直接卡在地板里
            min_pos[2] = min(min_pos[2], -0.5)

            grid_dx = float(self.mls_core.params.grid_dx)
            if grid_dx <= 1e-4: grid_dx = 0.1

            grid_size = (max_pos - min_pos) / grid_dx
            grid_dim = np.ceil(grid_size).astype(int) + 3
            grid_dim = np.clip(grid_dim, 16, 512)

            self.mls_core.params.grid_origin = wp.vec3(min_pos)
            self.mls_core.params.grid_dim = wp.vec3i(grid_dim[0], grid_dim[1], grid_dim[2])

            print(
                f"MLS Init: Grid Origin={min_pos}, Dim={grid_dim}, dx={grid_dx}, K={self.mls_core.params.bulk_modulus:.1e}")

            self.mls_particle_data = self.mls_core.initialize_particles(positions, velocities, masses)

            num_cells = int(grid_dim[0] * grid_dim[1] * grid_dim[2])

            self.grid_mass = wp.zeros(num_cells, dtype=float, device=self.device)
            self.grid_mom = wp.zeros(num_cells, dtype=wp.vec3, device=self.device)

            self.mls_core.params.collider_active = 1
            self.mls_core.params.collider_center = CUBE_CENTER
            self.mls_core.params.collider_half_size = wp.vec3(CUBE_FULL_EXTENTS) * 0.5
            self.mls_core.params.collider_friction = 0.05
            self._initialized = True
            return True

        except Exception as e:
            print(f"MLS Init Error: {e}")
            return False

    def _step_impl(self, state_in, state_out, dt, scratch):
        if not self._initialized:
            if not self.initialize_mls_particles(state_in): return

        # === CFL 自适应时间步长 ===
        K = self.mls_core.params.bulk_modulus
        rho = self.mls_core.params.density
        dx = self.mls_core.params.grid_dx

        c_sound = np.sqrt(K / rho)
        v_max_est = 20.0  # 估算最大速度

        # CFL Safety factor
        cfl_alpha = 0.4
        max_dt = cfl_alpha * dx / (c_sound + v_max_est + 1e-6)

        num_sub_steps = int(np.ceil(dt / max_dt))
        num_sub_steps = max(1, min(num_sub_steps, 500))  # 限制最大子步数

        sub_dt = dt / float(num_sub_steps)

        # 偶尔打印日志
        if (state_in.step if hasattr(state_in, 'step') else 0) % 60 == 0:
            print(f"Step info: dt={dt:.4f} -> {num_sub_steps} substeps of {sub_dt:.2e}s. c={c_sound:.1f}")

        for _ in range(num_sub_steps):
            self._sub_step(sub_dt)

        if hasattr(state_out, 'particle_q'):
            wp.copy(state_out.particle_q, self.mls_particle_data.position)
        if hasattr(state_out, 'particle_qd'):
            wp.copy(state_out.particle_qd, self.mls_particle_data.velocity)

    def _sub_step(self, dt):
        self.grid_mass.zero_()
        self.grid_mom.zero_()

        # 1. Update Stress
        wp.launch(
            update_particle_stress_kernel,
            dim=self.mls_particle_data.position.shape[0],
            inputs=[
                self.mls_particle_data,
                self.mls_core.params.viscosity,
                self.mls_core.params.bulk_modulus
            ],
            device=self.device
        )

        # 2. P2G
        wp.launch(
            kernel=mls_p2g_apic_kernel,
            dim=self.mls_particle_data.position.shape[0],
            inputs=[
                self.mls_particle_data,
                self.mls_core.params.grid_origin,
                self.mls_core.params.grid_dx,
                self.mls_core.params.grid_dim,
                dt,
                self.grid_mass,
                self.grid_mom
            ],
            device=self.device
        )

        # 3. Update Grid
        wp.launch(
            kernel=update_grid_kernel,
            dim=self.grid_mass.shape[0],
            inputs=[
                self.grid_mass,
                self.grid_mom,
                self.mls_core.params,
                dt
            ],
            device=self.device
        )

        # 4. G2P
        wp.launch(
            kernel=mls_g2p_apic_kernel,
            dim=self.mls_particle_data.position.shape[0],
            inputs=[
                self.mls_particle_data,
                self.grid_mass,
                self.grid_mom,
                self.mls_core.params.grid_origin,
                self.mls_core.params.grid_dx,
                self.mls_core.params.grid_dim,
                dt
            ],
            device=self.device
        )


@wp.kernel
def mls_p2g_apic_kernel(
        p_data: MLSMPMParticleData,
        grid_origin: wp.vec3,
        dx: float,
        grid_dim: wp.vec3i,
        dt: float,
        grid_mass: wp.array(dtype=float),
        grid_mom: wp.array(dtype=wp.vec3)
):
    p = wp.tid()
    pos = p_data.position[p]
    mass = p_data.mass[p]
    vel = p_data.velocity[p]
    C = p_data.C[p]
    stress = p_data.stress[p]
    volume = p_data.volume[p]

    inv_dx = 1.0 / dx
    pos_grid = (pos - grid_origin) * inv_dx
    base_pos = pos_grid - wp.vec3(0.5, 0.5, 0.5)
    base_idx = wp.vec3i(
        int(wp.floor(base_pos[0])),
        int(wp.floor(base_pos[1])),
        int(wp.floor(base_pos[2]))
    )
    fx = pos_grid - wp.vec3(float(base_idx[0]), float(base_idx[1]), float(base_idx[2]))

    w = wp.mat33(
        0.5 * (1.5 - fx[0]) * (1.5 - fx[0]), 0.75 - (fx[0] - 1.0) * (fx[0] - 1.0), 0.5 * (fx[0] - 0.5) * (fx[0] - 0.5),
        0.5 * (1.5 - fx[1]) * (1.5 - fx[1]), 0.75 - (fx[1] - 1.0) * (fx[1] - 1.0), 0.5 * (fx[1] - 0.5) * (fx[1] - 0.5),
        0.5 * (1.5 - fx[2]) * (1.5 - fx[2]), 0.75 - (fx[2] - 1.0) * (fx[2] - 1.0), 0.5 * (fx[2] - 0.5) * (fx[2] - 0.5)
    )

    term_scale = -4.0 * dt * volume * inv_dx * inv_dx
    stress_term = term_scale * stress
    affine_term = mass * C

    for i in range(3):
        for j in range(3):
            for k in range(3):
                ix = base_idx[0] + i
                iy = base_idx[1] + j
                iz = base_idx[2] + k

                # Check bounds strictly
                if ix >= 0 and ix < grid_dim[0] and iy >= 0 and iy < grid_dim[1] and iz >= 0 and iz < grid_dim[2]:
                    node_idx = ix * grid_dim[1] * grid_dim[2] + iy * grid_dim[2] + iz

                    offset = wp.vec3(float(i), float(j), float(k))
                    dpos = (offset - fx) * dx

                    weight = w[0, i] * w[1, j] * w[2, k]

                    momentum_contrib = weight * (mass * vel + affine_term @ dpos + stress_term @ dpos)

                    wp.atomic_add(grid_mass, node_idx, weight * mass)
                    wp.atomic_add(grid_mom, node_idx, momentum_contrib)


@wp.kernel
def update_grid_kernel(
        grid_mass: wp.array(dtype=float),
        grid_mom: wp.array(dtype=wp.vec3),
        params: MLSMPMParameters,
        dt: float
):
    idx = wp.tid()
    m = grid_mass[idx]

    if m > 1e-10:
        v = grid_mom[idx] / m
        v = v + params.gravity * dt

        grid_dim = params.grid_dim
        dim_yz = grid_dim[1] * grid_dim[2]
        i = idx // dim_yz
        rem = idx % dim_yz
        j = rem // grid_dim[2]
        k = rem % grid_dim[2]

        node_pos = params.grid_origin + wp.vec3(float(i), float(j), float(k)) * params.grid_dx
        if params.collider_active > 0:
            local_pos = node_pos - params.collider_center
            dist = sd_box(local_pos, params.collider_half_size)

            # dist < 0 表示在物体内部
            if dist < 0.0:
                normal = get_box_normal(local_pos, params.collider_half_size)

                v_in = v
                v_dot_n = wp.dot(v_in, normal)
                if v_dot_n < 0.0:
                    v_tangent = v_in - normal * v_dot_n

                    mu = params.collider_friction
                    v_tangent = v_tangent * wp.max(0.0, 1.0 - mu)

                    # 硬约束：法向速度归零（不做反弹，防止水乱飞）
                    v = v_tangent * 0

        bound = 2

        # 简单的 Sticky Boundary
        if k < bound and v[2] < 0.0: v = wp.vec3(0.0)
        if k > grid_dim[2] - bound and v[2] > 0.0: v = wp.vec3(v[0], v[1], 0.0)

        if i < bound and v[0] < 0.0: v = wp.vec3(0.0)
        if i > grid_dim[0] - bound and v[0] > 0.0: v = wp.vec3(0.0)

        if j < bound and v[1] < 0.0: v = wp.vec3(0.0)
        if j > grid_dim[1] - bound and v[1] > 0.0: v = wp.vec3(0.0)

        grid_mom[idx] = v * m


@wp.kernel
def mls_g2p_apic_kernel(
        p_data: MLSMPMParticleData,
        grid_mass: wp.array(dtype=float),
        grid_mom: wp.array(dtype=wp.vec3),
        grid_origin: wp.vec3,
        dx: float,
        grid_dim: wp.vec3i,
        dt: float
):
    p = wp.tid()
    pos = p_data.position[p]

    inv_dx = 1.0 / dx
    pos_grid = (pos - grid_origin) * inv_dx
    base_pos = pos_grid - wp.vec3(0.5, 0.5, 0.5)
    base_idx = wp.vec3i(
        int(wp.floor(base_pos[0])),
        int(wp.floor(base_pos[1])),
        int(wp.floor(base_pos[2]))
    )
    fx = pos_grid - wp.vec3(float(base_idx[0]), float(base_idx[1]), float(base_idx[2]))

    w = wp.mat33(
        0.5 * (1.5 - fx[0]) * (1.5 - fx[0]), 0.75 - (fx[0] - 1.0) * (fx[0] - 1.0), 0.5 * (fx[0] - 0.5) * (fx[0] - 0.5),
        0.5 * (1.5 - fx[1]) * (1.5 - fx[1]), 0.75 - (fx[1] - 1.0) * (fx[1] - 1.0), 0.5 * (fx[1] - 0.5) * (fx[1] - 0.5),
        0.5 * (1.5 - fx[2]) * (1.5 - fx[2]), 0.75 - (fx[2] - 1.0) * (fx[2] - 1.0), 0.5 * (fx[2] - 0.5) * (fx[2] - 0.5)
    )

    new_v = wp.vec3(0.0)
    new_C = wp.mat33(0.0)

    for i in range(3):
        for j in range(3):
            for k in range(3):
                ix = base_idx[0] + i
                iy = base_idx[1] + j
                iz = base_idx[2] + k

                if ix >= 0 and ix < grid_dim[0] and iy >= 0 and iy < grid_dim[1] and iz >= 0 and iz < grid_dim[2]:
                    node_idx = ix * grid_dim[1] * grid_dim[2] + iy * grid_dim[2] + iz

                    m = grid_mass[node_idx]
                    if m > 1e-10:
                        offset = wp.vec3(float(i), float(j), float(k))
                        dpos = (offset - fx) * dx
                        weight = w[0, i] * w[1, j] * w[2, k]

                        node_v = grid_mom[node_idx] / m
                        new_v += weight * node_v
                        new_C += 4.0 * weight * wp.outer(node_v, dpos) * (inv_dx * inv_dx)

    p_data.velocity[p] = new_v
    p_data.position[p] += new_v * dt
    p_data.C[p] = new_C

    # 更新 J
    J_old = p_data.J[p]
    # 使用 1+dt*tr(C) 近似
    J_new = J_old * (1.0 + dt * (new_C[0, 0] + new_C[1, 1] + new_C[2, 2]))
    p_data.J[p] = J_new

    # 重置弹性形变 (Fluid)
    p_data.deformation_gradient[p] = wp.identity(n=3, dtype=float)
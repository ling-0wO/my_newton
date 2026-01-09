import warp as wp
import numpy as np
import sys
import os
import newton

from newton._src.solvers.mls_mpm.solver_mls_mpm import SolverMLSMPM, MLSMPMOptions
from newton._src.solvers.mls_mpm.mls_types import MLSMPMParticleData

class MockState:
    def __init__(self, pos, vel):
        self.particle_q = wp.array(pos, dtype=wp.vec3)
        self.particle_qd = wp.array(vel, dtype=wp.vec3)
        self.body_q = None
        self.body_qd = None
        self.step = 0


def run_mass_test():
    wp.init()
    print("--- 开始质量守恒测试 (Mass Conservation Test) ---")

    builder = newton.ModelBuilder()

    voxel_size = 0.05
    density = 1000.0

    particle_lo = np.array([-0.4, -0.4, 0.5])
    particle_hi = np.array([0.4, 0.4, 0.9])

    particles_per_cell = 4
    particle_res = np.array(
        np.ceil(particles_per_cell * (particle_hi - particle_lo) / voxel_size),
        dtype=int,
    )
    cell_size = (particle_hi - particle_lo) / particle_res
    cell_volume = np.prod(cell_size)
    radius = np.max(cell_size) * 0.5
    mass = np.prod(cell_volume) * density

    print(f"生成粒子网格: Res={particle_res}")

    builder.add_particle_grid(
        pos=wp.vec3(particle_lo),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0),
        dim_x=particle_res[0],
        dim_y=particle_res[1],
        dim_z=particle_res[2],
        cell_x=cell_size[0],
        cell_y=cell_size[1],
        cell_z=cell_size[2],
        mass=mass,
        jitter=0.0,
        radius_mean=radius,
    )

    builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=0.5))

    model = builder.finalize()
    model.particle_mu = 0.0
    model.particle_ke = 0.0
    model.particle_kd = 0.0
    model.particle_cohesion = 0.0
    model.particle_adhesion = 0.0

    print(f"模型构建完成。粒子数量: {model.particle_count}")

    options = MLSMPMOptions()
    options.grid_dx = voxel_size
    options.support_radius = voxel_size * 1.5
    options.fluid_density = density
    options.viscosity = 0.0
    options.bulk_modulus = 1.0e5

    print("正在初始化 MLS 求解器...")

    solver = SolverMLSMPM(model, options)
    print("✅ 求解器初始化成功！")

    state_0 = model.state()

    solver.initialize_mls_particles(state_0)

    wp.synchronize()
    if solver.mls_particle_data is None:
        print("❌ 错误: MLS 粒子数据未初始化")
        return

    initial_total_mass = wp.utils.array_sum(solver.mls_particle_data.mass)
    print(f"初始总质量 (T=0): {initial_total_mass:.6f}")

    dt = 0.01
    total_steps = 1000

    state_1 = model.state()

    for i in range(total_steps):

        solver._step_impl(state_0, state_1, dt, None, None)

        wp.synchronize()
        current_total_mass = wp.utils.array_sum(solver.mls_particle_data.mass)

        if np.isnan(current_total_mass):
            print(f"❌ 失败: 第 {i} 帧爆炸 (NaN)")
            return

        diff = abs(current_total_mass - initial_total_mass)

        if i % 10 == 0:
            print(f"Step {i:03d} | Mass: {current_total_mass:.6f} | Diff: {diff:.2e}")

    print("-" * 30)
    print(f"✅ 测试通过: {total_steps} 步后质量偏差仅为 {diff:.2e}")


if __name__ == "__main__":
    run_mass_test()
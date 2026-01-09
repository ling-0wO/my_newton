import numpy as np
import warp as wp
import sys
import os
import matplotlib.pyplot as plt


import newton
import newton.examples
from newton.solvers import SolverImplicitMPM
from newton._src.solvers.mls_mpm.solver_mls_mpm import SolverMLSMPM, MLSMPMOptions
from newton._src.solvers.mls_mpm.mls_types import MLSMPMParticleData


def run_simulation(viscosity_value, steps=300):
    """运行一次模拟并返回动能历史数据"""
    print(f"--- Running Simulation with Viscosity = {viscosity_value} ---")

    # 1. 配置
    builder = newton.ModelBuilder()
    voxel_size = 0.05
    density = 1000.0

    # 生成一个处于“不平衡状态”的水块 (例如左高右低，或者直接给初速度)
    # 这里我们生成一个悬空的水块，让它砸下来晃荡
    dam_lo = np.array([-0.3, -0.3, 0.5])
    dam_hi = np.array([0.3, 0.3, 1.0])

    particles_per_cell = 5
    particle_res = np.array(np.ceil(particles_per_cell * (dam_hi - dam_lo) / voxel_size), dtype=int)
    cell_size = (dam_hi - dam_lo) / particle_res
    cell_volume = np.prod(cell_size)
    mass = np.prod(cell_volume) * density

    builder.add_particle_grid(
        pos=wp.vec3(dam_lo),
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
        radius_mean=voxel_size * 0.25,
    )

    # 容器边界 (摩擦设为0，排除边界摩擦干扰，纯看内部粘性)
    builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=0.0))

    model = builder.finalize()
    model.set_gravity(wp.vec3(0.0, 0.0, -9.8))

    # 填充空参数
    model.particle_mu = 0.0
    model.particle_ke = 0.0
    model.particle_kd = 0.0
    model.particle_cohesion = 0.0
    model.particle_adhesion = 0.0

    # 2. Solver 选项
    options = MLSMPMOptions()
    options.grid_dx = voxel_size
    options.support_radius = voxel_size * 1.5
    options.fluid_density = density
    options.bulk_modulus = 1.0e5

    # 【关键变量】
    options.viscosity = viscosity_value

    solver = SolverMLSMPM(model, options)
    state_0 = model.state()
    state_1 = model.state()
    solver.initialize_mls_particles(state_0)

    # 3. 循环与数据采集
    kinetic_energies = []
    dt = 0.016  # 60FPS

    for i in range(steps):
        solver._step_impl(state_0, state_1, dt, None, None)

        # 计算动能 Ek = 0.5 * sum(m * v^2)
        # 获取速度和质量
        wp.synchronize()
        # 注意：这里直接读取 mls_particle_data，它是在GPU上的
        # 我们需要在 Python 端计算动能，或者写个简单的 Kernel
        # 为了简单，转到 numpy 计算 (性能稍慢但测试够用)
        vel = solver.mls_particle_data.velocity.numpy()
        mass = solver.mls_particle_data.mass.numpy()

        v_sq = np.sum(vel ** 2, axis=1)
        ek = 0.5 * np.sum(mass * v_sq)

        kinetic_energies.append(ek)

        # 交换
        # state_0, state_1 = state_1, state_0 # 你的solver自更新，但这行保留无害

        if i % 50 == 0:
            print(f"Step {i}/{steps}, Ek={ek:.2f}")

    return kinetic_energies


def run_viscosity_test():
    wp.init()

    # 运行两组对照实验
    steps = 200
    ek_low = run_simulation(viscosity_value=0.0, steps=steps)
    ek_high = run_simulation(viscosity_value=50.0, steps=steps)  # 5.0 是非常高的粘度，像蜂蜜/沥青

    # 绘图
    plt.figure(figsize=(10, 6))
    x = np.arange(steps)

    plt.plot(x, ek_low, label='Viscosity = 0.0 (Water-like)', alpha=0.8)
    plt.plot(x, ek_high, label='Viscosity = 5.0 (Honey-like)', linewidth=2)

    plt.title('Kinetic Energy Decay Test')
    plt.xlabel('Simulation Steps')
    plt.ylabel('Total Kinetic Energy (J)')
    plt.legend()
    plt.grid(True)

    output_file = 'test_viscosity_result.png'
    plt.savefig(output_file)
    print(f"✅ 测试完成！结果已保存为 {output_file}")
    print("预期结果：高粘度曲线（橙色）的能量应该比低粘度（蓝色）衰减得更快。")
    # plt.show()


if __name__ == "__main__":
    run_viscosity_test()
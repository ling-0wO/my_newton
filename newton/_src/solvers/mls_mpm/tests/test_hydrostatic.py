import warp as wp
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import newton

from newton._src.solvers.mls_mpm.solver_mls_mpm import SolverMLSMPM, MLSMPMOptions
from newton._src.solvers.mls_mpm.mls_types import MLSMPMParticleData


def run_hydrostatic_test():
    wp.init()
    print("--- 开始静水压力测试 (Hydrostatic Pressure Test) ---")

    bulk_modulus = 1.0e4
    fluid_density = 1000.0
    gamma = 7.0
    voxel_size = 0.05
    gravity = -9.8

    water_height = 1.5

    builder = newton.ModelBuilder()

    particle_lo = np.array([-0.2, -0.2, 0.5])
    particle_hi = np.array([0.2, 0.2, 2.0])

    particles_per_cell = 4
    particle_res = np.array(
        np.ceil(particles_per_cell * (particle_hi - particle_lo) / voxel_size),
        dtype=int,
    )
    cell_size = (particle_hi - particle_lo) / particle_res
    cell_volume = np.prod(cell_size)
    mass = np.prod(cell_volume) * fluid_density

    print(f"生成水柱: 高度={water_height}m, 粒子层数={particle_res[2]}")

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
        radius_mean=voxel_size * 0.25,
    )

    # 添加容器底部和四壁 (防止水散开)
    # 地面
    builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=0.0))  # 无摩擦最好

    model = builder.finalize()
    model.set_gravity(wp.vec3(0.0, 0.0, gravity))

    # 填充父类依赖参数
    model.particle_mu = 0.0
    model.particle_ke = 0.0
    model.particle_kd = 0.0
    model.particle_cohesion = 0.0
    model.particle_adhesion = 0.0

    options = MLSMPMOptions()
    options.grid_dx = voxel_size
    options.support_radius = voxel_size * 1.5
    options.fluid_density = fluid_density
    options.viscosity = 0.1
    options.bulk_modulus = bulk_modulus

    solver = SolverMLSMPM(model, options)
    state_0 = model.state()
    state_1 = model.state()
    solver.initialize_mls_particles(state_0)


    dt = 0.005
    total_time = 5.0
    steps = int(total_time / dt)

    print(f"正在模拟沉降过程 ({steps} 步)...")

    for i in range(steps):
        solver._step_impl(state_0, state_1, dt, None, None)

        if i % 5 == 0:
            vel = solver.mls_particle_data.velocity.numpy()
            vel *= 0.99
            solver.mls_particle_data.velocity = wp.array(vel, dtype=wp.vec3, device=solver.device)

        if i % 50 == 0:
            print(f"进度: {i}/{steps}")

    wp.synchronize()

    pos = solver.mls_particle_data.position.numpy()
    J = solver.mls_particle_data.J.numpy()

    J_clamped = np.clip(J, 0.1, 1.05)
    pressures_sim = bulk_modulus * (np.power(J_clamped, -gamma) - 1.0)
    pressures_sim[pressures_sim < -1000.0] = -1000.0
    mask = (np.abs(pos[:, 0]) < 0.1) & (np.abs(pos[:, 1]) < 0.1)
    z_valid = pos[mask, 2]
    p_valid = pressures_sim[mask]


    water_surface_height = np.max(z_valid)
    print(f"最终水面高度: {water_surface_height:.4f} m")

    pressures_theory = -fluid_density * gravity * (water_surface_height - z_valid)
    pressures_theory = np.maximum(pressures_theory, 0)  # 压力不能为负

    plt.figure(figsize=(10, 6))

    plt.scatter(p_valid, z_valid, s=5, alpha=0.3, label='MPM Particles', color='blue')


    z_line = np.linspace(0, water_surface_height, 100)
    p_line = -fluid_density * gravity * (water_surface_height - z_line)
    plt.plot(p_line, z_line, 'r--', linewidth=3, label='Theoretical: P = ρg(H-h)')

    plt.ylabel('Height Z (m)')
    plt.xlabel('Pressure P (Pa)')
    plt.title(f'Hydrostatic Pressure Test\nBulk Modulus={bulk_modulus:.0e}, Density={fluid_density}')
    plt.legend()
    plt.grid(True)

    output_file = 'test_hydrostatic_result.png'
    plt.savefig(output_file)
    print(f"✅ 测试完成！结果图表已保存至: {output_file}")
    print("如果红色虚线（理论值）穿过蓝色点云（模拟值）的中心，则通过测试。")


if __name__ == "__main__":
    run_hydrostatic_test()
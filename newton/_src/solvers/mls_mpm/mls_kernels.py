import warp as wp
import numpy as np
from .mls_types import MLSMPMParticleData

@wp.func
def sd_box(p: wp.vec3, b: wp.vec3):
    d = wp.vec3(wp.abs(p[0]) - b[0], wp.abs(p[1]) - b[1], wp.abs(p[2]) - b[2])
    outside = wp.length(wp.max(d, wp.vec3(0.0)))
    inside = wp.min(wp.max(d[0], wp.max(d[1], d[2])), 0.0)
    return outside + inside

@wp.func
def get_box_normal(p: wp.vec3, b: wp.vec3):
    dx = wp.abs(p[0]) - b[0]
    dy = wp.abs(p[1]) - b[1]
    dz = wp.abs(p[2]) - b[2]

    if dx > dy and dx > dz:
        return wp.vec3(wp.sign(p[0]), 0.0, 0.0)
    elif dy > dz:
        return wp.vec3(0.0, wp.sign(p[1]), 0.0)
    else:
        return wp.vec3(0.0, 0.0, wp.sign(p[2]))

@wp.func
def fluid_constitutive_model(J: float, C: wp.mat33, deformation_grad: wp.mat33, viscosity: float,
                             bulk_modulus: float) -> wp.mat33:
    gamma = 7.0
    # 限制 J 的范围，防止 NaN，但允许一定的压缩
    J_clamped = wp.clamp(J, 0.1, 1.05)
    pressure = bulk_modulus * (wp.pow(J_clamped, -gamma) - 1.0)

    # 消除负压
    if pressure < -1000.0:
        #print(pressure)
        pressure = -1000.0

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

                # 边界检测
                if ix >= 0 and ix < grid_dim[0] and iy >= 0 and iy < grid_dim[1] and iz >= 0 and iz < grid_dim[2]:
                    node_idx = ix * grid_dim[1] * grid_dim[2] + iy * grid_dim[2] + iz

                    offset = wp.vec3(float(i), float(j), float(k))
                    dpos = (offset - fx) * dx

                    weight = w[0, i] * w[1, j] * w[2, k]

                    momentum_contrib = weight * (mass * vel + affine_term @ dpos + stress_term @ dpos)

                    wp.atomic_add(grid_mass, node_idx, weight * mass)
                    wp.atomic_add(grid_mom, node_idx, momentum_contrib)


@wp.kernel
def update_grid_with_coupling_kernel(
        grid_mass: wp.array(dtype=float),
        grid_mom: wp.array(dtype=wp.vec3),
        body_forces_out: wp.array(dtype=wp.spatial_vector),
        gravity: wp.vec3,
        dt: float,
        grid_dim: wp.vec3i,
        grid_origin: wp.vec3,
        grid_dx: float,
        num_bodies: int,
        body_q: wp.array(dtype=wp.transform),
        body_qd: wp.array(dtype=wp.spatial_vector),
        body_sizes: wp.array(dtype=wp.vec3),
        friction: float,
        feedback_scale: float,
        shape_type: int
):
    idx = wp.tid()
    m = grid_mass[idx]

    # 质量截断
    if m <= 1e-6:
        grid_mom[idx] = wp.vec3(0.0)
        return

    v_in = grid_mom[idx] / m
    v_pred = v_in + gravity * dt

    # 坐标计算
    dim_yz = grid_dim[1] * grid_dim[2]
    i = idx // dim_yz
    rem = idx % dim_yz
    j = rem // grid_dim[2]
    k = rem % grid_dim[2]

    if i < 0 or i >= grid_dim[0] or j < 0 or j >= grid_dim[1] or k < 0 or k >= grid_dim[2]:
        return

    pos_world = grid_origin + wp.vec3(float(i), float(j), float(k)) * grid_dx

    # 简单的地面碰撞处理 (z=0)
    if pos_world[2] <= 0.001 and v_pred[2] < 0.0:
        v_pred = wp.vec3(v_pred[0], v_pred[1], 0.0)

    if shape_type:
        X_wb = body_q[0]
        half_size = body_sizes[0]
        radius = half_size[0]
        pos_body = wp.transform_point(wp.transform_inverse(X_wb), pos_world)
        len_p = wp.length(pos_body)
        dist = len_p - radius

        if len_p > 1e-6:
            normal_body = pos_body / len_p
        else:
            normal_body = wp.vec3(0.0, 0.0, 1.0)
    # 多刚体循环
    else:
        if num_bodies > 0:
            for b_idx in range(num_bodies):
                X_wb = body_q[b_idx]
                v_spatial = body_qd[b_idx]
                half_size = body_sizes[b_idx]

                pos_body = wp.transform_point(wp.transform_inverse(X_wb), pos_world)

                # SDF Box
                d = wp.vec3(wp.abs(pos_body[0]), wp.abs(pos_body[1]), wp.abs(pos_body[2])) - half_size
                dist = wp.length(wp.max(d, wp.vec3(0.0))) + wp.min(wp.max(d[0], wp.max(d[1], d[2])), 0.0)

                margin = grid_dx * 0.5
                if dist < margin:
                    weight = wp.clamp((margin - dist) / margin, 0.0, 1.0)
                    normal_body = wp.vec3(0.0)
                    if d[0] > d[1] and d[0] > d[2]:
                        normal_body = wp.vec3(wp.sign(pos_body[0]), 0.0, 0.0)
                    elif d[1] > d[2]:
                        normal_body = wp.vec3(0.0, wp.sign(pos_body[1]), 0.0)
                    else:
                        normal_body = wp.vec3(0.0, 0.0, wp.sign(pos_body[2]))

                    normal_world = wp.quat_rotate(wp.transform_get_rotation(X_wb), normal_body)

                    v_lin = wp.vec3(v_spatial[0], v_spatial[1], v_spatial[2])
                    v_ang = wp.vec3(v_spatial[3], v_spatial[4], v_spatial[5])

                    body_pos_world = wp.transform_get_translation(X_wb)
                    r_world = pos_world - body_pos_world

                    v_body = v_lin + wp.cross(v_ang, r_world)

                    v_rel = v_pred - v_body
                    v_rel_n = wp.dot(v_rel, normal_world)

                    if v_rel_n < 0.0:
                        v_target = v_body + (v_rel - normal_world * v_rel_n)
                        delta_v = (v_target - v_pred) * weight

                        v_pred = v_pred + delta_v

                        impulse = -delta_v * m * feedback_scale

                        torque_impulse = wp.cross(r_world, impulse)

                        torque_len = wp.length(torque_impulse)
                        if torque_len > 10.0 * dt:
                            torque_impulse = torque_impulse / torque_len * (10.0 * dt)

                        f_vec = impulse / dt
                        t_vec = torque_impulse / dt

                        wp.atomic_add(body_forces_out, b_idx, wp.spatial_vector(f_vec, t_vec))

    # 边界条件
    boundary_width = 2
    if k < boundary_width and v_pred[2] < 0.0: v_pred = wp.vec3(v_pred[0], v_pred[1], 0.0)
    if k > grid_dim[2] - boundary_width and v_pred[2] > 0.0: v_pred = wp.vec3(v_pred[0], v_pred[1], 0.0)
    if i < boundary_width and v_pred[0] < 0.0: v_pred = wp.vec3(0.0, v_pred[1], v_pred[2])
    if i > grid_dim[0] - boundary_width and v_pred[0] > 0.0: v_pred = wp.vec3(0.0, v_pred[1], v_pred[2])
    if j < boundary_width and v_pred[1] < 0.0: v_pred = wp.vec3(v_pred[0], 0.0, v_pred[2])
    if j > grid_dim[1] - boundary_width and v_pred[1] > 0.0: v_pred = wp.vec3(v_pred[0], 0.0, v_pred[2])

    v_max = 20.0
    v_pred = wp.vec3(
        wp.clamp(v_pred[0], -v_max, v_max),
        wp.clamp(v_pred[1], -v_max, v_max),
        wp.clamp(v_pred[2], -v_max, v_max)
    )

    grid_mom[idx] = v_pred * m

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

    J_old = p_data.J[p]
    div_v = new_C[0, 0] + new_C[1, 1] + new_C[2, 2]

    # 限制每步体积变化率
    clamped_div = wp.clamp(div_v, -0.05 / dt, 0.05 / dt)
    J_new = J_old * (1.0 + dt * clamped_div)
    p_data.J[p] = J_new

    # 重置弹性形变
    p_data.deformation_gradient[p] = wp.identity(n=3, dtype=float)
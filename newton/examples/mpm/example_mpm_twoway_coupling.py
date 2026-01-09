# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import warp as wp
import newton
import newton.examples
from newton.solvers import SolverMLSMPM

# === 强制清理缓存 ===
try:
    wp.config.clear_kernel_cache()
except:
    pass


@wp.kernel
def compute_body_forces(
        dt: float,
        collider_ids: wp.array(dtype=int),
        collider_impulses: wp.array(dtype=wp.vec3),
        collider_impulse_pos: wp.array(dtype=wp.vec3),
        body_ids: wp.array(dtype=int),
        body_q: wp.array(dtype=wp.transform),
        body_com: wp.array(dtype=wp.vec3),
        body_f: wp.array(dtype=wp.spatial_vector),
):
    i = wp.tid()
    cid = collider_ids[i]

    # [关键修复] 严格检查索引有效性
    # 1. 映射表索引检查
    if cid < 0 or cid >= body_ids.shape[0]:
        return

    body_index = body_ids[cid]

    # 2. 刚体索引检查 (防止访问 body_q[0] 当 body_q 为空时)
    if body_index < 0 or body_index >= body_q.shape[0]:
        return

    f_val = collider_impulses[i] / dt

    # [安全限幅]
    limit = 1000.0
    if wp.length_sq(f_val) > limit * limit:
        f_val = wp.normalize(f_val) * limit

    X_wb = body_q[body_index]
    X_com = body_com[body_index]
    r = collider_impulse_pos[i] - wp.transform_point(X_wb, X_com)
    wp.atomic_add(body_f, body_index, wp.spatial_vector(f_val, wp.cross(r, f_val)))


class Example:
    def __init__(self, viewer, options=None):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.viewer = viewer

        # 1. 场景: 纯流体
        builder = newton.ModelBuilder()
        self._emit_particles(builder)
        builder.add_ground_plane()

        # [注意] 这里没有 add_body，所以 body_count = 0

        self.model = builder.finalize()
        self.model.set_gravity(np.array((0.0, 0.0, -9.8)))
        self.control = self.model.control()

        # 2. Solver 配置
        mpm_options = SolverMLSMPM.Options()
        mpm_options.bulk_modulus = 1.0e6
        # 禁用碰撞体，既然没有刚体，就不需要计算耦合
        mpm_options.collider_active = 0
        mpm_options.collider_half_size = wp.vec3(0.5)

        self.solver_mpm = SolverMLSMPM(self.model, mpm_options)
        self.solver_rb = newton.solvers.SolverXPBD(self.model)

        self.state = self.model.state()
        self.solver_mpm.enrich_state(self.state)

        self.viewer.set_model(self.model)
        self.viewer.show_particles = True

        # 3. 预分配 Buffer
        # 匹配 Solver 中的 60x60x60 (216000)
        # 如果 Solver 里写的是 40x40x40 (64000)
        # 为了安全，分配大一点
        max_nodes = 256 * 256 * 10
        self.collider_impulses = wp.zeros(max_nodes, dtype=wp.vec3, device=self.model.device)
        self.collider_impulse_pos = wp.zeros(max_nodes, dtype=wp.vec3, device=self.model.device)
        self.collider_impulse_ids = wp.full(max_nodes, value=-1, dtype=int, device=self.model.device)

        # 这里的 map 是空的，因为没有刚体
        self.collider_body_map = wp.zeros(1, dtype=int, device=self.model.device)

    def _emit_particles(self, builder):
        count = 10000
        radius = 0.05
        rng = np.random.default_rng(42)
        points = rng.uniform(low=[-0.5, -0.5, 0.2], high=[0.5, 0.5, 0.8], size=(count, 3))

        builder.particle_q = points
        builder.particle_qd = np.zeros_like(points)
        builder.particle_mass = np.full(count, 1.0)
        builder.particle_radius = np.full(count, radius)
        builder.particle_flags = np.ones(count, dtype=int)

    def simulate(self):
        for _ in range(self.sim_substeps):
            # A. 没有刚体，无需同步

            # B. MPM Step
            self.solver_mpm.step(self.state, self.state, None, None, self.sim_dt)

            # C. 双向耦合路径
            # 只有当存在刚体时，才执行力回传，否则会越界访问 body_q
            if self.model.body_count > 0:
                impulses, pos, ids = self.solver_mpm.collect_collider_impulses(self.state)

                n = min(impulses.shape[0], self.collider_impulses.shape[0])
                if n > 0:
                    wp.copy(self.collider_impulses, impulses, count=n)
                    wp.copy(self.collider_impulse_pos, pos, count=n)
                    wp.copy(self.collider_impulse_ids, ids, count=n)

                    self.state.clear_forces()

                    wp.launch(
                        compute_body_forces,
                        dim=n,
                        inputs=[
                            self.sim_dt,
                            self.collider_impulse_ids,
                            self.collider_impulses,
                            self.collider_impulse_pos,
                            self.collider_body_map,
                            self.state.body_q,
                            self.model.body_com,
                            self.state.body_f
                        ],
                        device=self.model.device
                    )

            # D. XPBD Step (即使没刚体也要跑，可能有一些默认逻辑)
            contacts = self.model.collide(self.state)
            self.solver_rb.step(self.state, self.state, self.control, contacts, self.sim_dt)

    def step(self):
        self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state)
        self.viewer.end_frame()


if __name__ == "__main__":
    viewer, args = newton.examples.init()
    example = Example(viewer, args)
    newton.examples.run(example, args)
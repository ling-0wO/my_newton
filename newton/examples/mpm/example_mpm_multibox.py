# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.solvers import SolverImplicitMPM
from newton.solvers import SolverMLSMPM
from newton.solvers import SolverXPBD
CUBE_FULL_EXTENTS = (0.5, 0.5, 0.5)
CUBE_CENTER = (0.25, 0.25, 0.25)
cube_mass = 20.0  # 5, 20, 50

@wp.kernel
def compute_body_forces(
        dt: float,
        grid_impulses: wp.array(dtype=wp.vec3),
        body_f: wp.array(dtype=wp.spatial_vector),
        grid_origin: wp.vec3,
        grid_dx: float,
        grid_dim: wp.vec3i,
        body_q: wp.array(dtype=wp.transform),
        body_com: wp.array(dtype=wp.vec3)
):
    idx = wp.tid()
    impulse = grid_impulses[idx]
    if wp.length_sq(impulse) > 1e-12:
        f_val = impulse / dt

        # 计算位置
        dim_yz = grid_dim[1] * grid_dim[2]
        i = idx // dim_yz
        rem = idx % dim_yz
        j = rem // grid_dim[2]
        k = rem % grid_dim[2]
        pos_world = grid_origin + wp.vec3(float(i), float(j), float(k)) * grid_dx

        X_wb = body_q[0]
        pos_com_world = wp.transform_point(X_wb, body_com[0])
        r = pos_world - pos_com_world
        torque = wp.cross(r, f_val)

        torque_limit = 50.0
        if wp.length(torque) > torque_limit:
            torque = wp.normalize(torque) * torque_limit

        wp.atomic_add(body_f, 0, wp.spatial_vector(f_val, torque))

class Example:
    def __init__(self, viewer, options):
        # setup simulation parameters first
        self.fps = options.fps
        self.frame_dt = 1.0 / self.fps

        # group related attributes by prefix
        self.sim_time = 0.0
        self.sim_substeps = options.substeps
        self.sim_dt = self.frame_dt / self.sim_substeps

        # save a reference to the viewer
        self.viewer = viewer
        builder = newton.ModelBuilder()
        Example.emit_particles(builder, options)

        self.rigid_bodies_config = [
            {'pos': (-0.4,-0.4, 2.0), 'mass': 200.0, 'size': (0.5, 0.5, 0.5)},
            {'pos': (0.4, 0.4, 2.5), 'mass': 22.5, 'size': (0.35, 0.35, 0.35)},
            {'pos': (0, 0, 3.0), 'mass': 1.0, 'size': (0.2, 0.2, 0.2)}
        ]

        num_bodies = len(self.rigid_bodies_config)

        # 预分配平滑力数组
        self.smoothed_body_f = np.zeros((num_bodies, 6), dtype=np.float32)

        # Setup collision geometry
        self.collider = options.collider
        if self.collider == "concave":
            extents = (1.0, 2.0, 0.25)
            left_xform = wp.transform(
                wp.vec3(-0.7, 0.0, 0.8), wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), np.pi / 4.0)
            )
            right_xform = wp.transform(
                wp.vec3(0.7, 0.0, 0.8), wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), -np.pi / 4.0)
            )

            builder.add_shape_box(
                body=-1,
                cfg=newton.ModelBuilder.ShapeConfig(mu=0.1, density=0.0),
                xform=left_xform,
                hx=extents[0],
                hy=extents[1],
                hz=extents[2],
            )
            builder.add_shape_box(
                body=-1,
                cfg=newton.ModelBuilder.ShapeConfig(mu=0.1, density=0.0),
                xform=right_xform,
                hx=extents[0],
                hy=extents[1],
                hz=extents[2],
            )

        if self.collider == "cube":
            hidden_pose = wp.transform(wp.vec3(999.0, 999.0, 999.0), wp.quat_identity())
            self.box_spawn_time = 2.0
            self.box_spawned = False
            self.filter_alpha = 0.15
            self.smoothed_body_f = np.zeros((num_bodies, 6), dtype=np.float32)
            for cfg in self.rigid_bodies_config:
                body_id = builder.add_body(
                    mass=cfg['mass'],
                    xform=hidden_pose
                )

                builder.add_shape_box(
                    body=body_id,
                    cfg=newton.ModelBuilder.ShapeConfig(mu=0.5, density=0.0),
                    hx=cfg['size'][0],
                    hy=cfg['size'][1],
                    hz=cfg['size'][2],
                    xform=wp.transform_identity()
                )

        elif self.collider == "wedge":
            xform = wp.transform(
                wp.vec3(0.0, 0.0, 0.9), wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), np.pi / 4.0)
            )
            builder.add_shape_box(body=-1, cfg=newton.ModelBuilder.ShapeConfig(mu=0.1), xform=xform, hx=0.5, hy=2.0,
                                      hz=0.8)

            builder.add_shape_box(
                body=-1,
                cfg=newton.ModelBuilder.ShapeConfig(mu=0.1),
                xform=xform,
                hx=extents[0],
                hy=extents[1],
                hz=extents[2],
            )

        builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=0.5))

        self.model = builder.finalize()
        self.model.particle_mu = options.friction_coeff
        self.model.set_gravity(options.gravity)

        self.model.particle_ke = 0.0
        self.model.particle_kd = 0.0
        self.model.particle_cohesion = 0.0
        self.model.particle_adhesion = 0.0

        # Copy all remaining CLI arguments to MPM options
        mpm_options = SolverMLSMPM.Options()
        mpm_options.support_radius = 0.05
        mpm_options.viscosity = 0.01
        mpm_options.fluid_density = 1000.0
        mpm_options.bulk_modulus = 1.0e4
        mpm_options.collider_active = 1
        for key in vars(options):
            if hasattr(mpm_options, key):
                setattr(mpm_options, key, getattr(options, key))

        # Create MPM model from Newton model
        mpm_model = SolverImplicitMPM.Model(self.model, mpm_options)
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()

        # Initialize MPM solver and add supplemental state variables
        self.solver = SolverMLSMPM(self.model, mpm_options)

        self.solver.enrich_state(self.state_0)
        self.solver.enrich_state(self.state_1)

        sizes_list = [wp.vec3(cfg['size']) for cfg in self.rigid_bodies_config]
        self.solver.body_sizes_buffer = wp.array(sizes_list, dtype=wp.vec3, device=self.model.device)

        self.viewer.set_model(self.model)
        # 初始化刚体求解器
        self.solver_rb = SolverXPBD(self.model)

        self.solver.enrich_state(self.state_0)
        if isinstance(self.viewer, newton.viewer.ViewerGL):
            self.viewer.register_ui_callback(self.render_ui, position="side")

        self.viewer.show_particles = True
        self.show_normals = False

        self.capture()

    def capture(self):
        self.graph = None
        if wp.get_device().is_cuda and self.solver.grid_type == "fixed":
            if self.sim_substeps % 2 != 0:
                wp.utils.warn("Sim substeps must be even for graph capture of MPM step")
            else:
                with wp.ScopedCapture() as capture:
                    self.simulate()
                self.graph = capture.graph

    def simulate(self):
        if not self.box_spawned and self.sim_time >= self.box_spawn_time:
            print(f"--- Spawning {len(self.rigid_bodies_config)} Boxes ---")

            new_pos_list = []
            new_vel_list = []
            inv_mass_list = []
            inv_inertia_list = []

            for cfg in self.rigid_bodies_config:
                pos = wp.transform(wp.vec3(*cfg['pos']), wp.quat_identity())
                new_pos_list.append(pos)

                wake_vel = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, -2.0)
                new_vel_list.append(wake_vel)

                m = cfg['mass']
                inv_mass_list.append(1.0 / m)

                hx, hy, hz = cfg['size']

                ixx = (m / 3.0) * (hy * hy + hz * hz)
                iyy = (m / 3.0) * (hx * hx + hz * hz)
                izz = (m / 3.0) * (hx * hx + hy * hy)

                inv_inertia = wp.mat33(
                    1.0 / ixx, 0.0, 0.0,
                    0.0, 1.0 / iyy, 0.0,
                    0.0, 0.0, 1.0 / izz
                )
                inv_inertia_list.append(inv_inertia)
            num = len(self.rigid_bodies_config)

            # 位置
            pos_arr = wp.array(new_pos_list, dtype=wp.transform, device=self.model.device)
            wp.copy(dest=self.state_0.body_q, src=pos_arr, count=num)
            wp.copy(dest=self.state_1.body_q, src=pos_arr, count=num)

            # 速度
            vel_arr = wp.array(new_vel_list, dtype=wp.spatial_vector, device=self.model.device)
            wp.copy(dest=self.state_0.body_qd, src=vel_arr, count=num)
            wp.copy(dest=self.state_1.body_qd, src=vel_arr, count=num)

            # 质量属性 (直接写回 model)
            mass_arr = wp.array(inv_mass_list, dtype=float, device=self.model.device)
            wp.copy(dest=self.model.body_inv_mass, src=mass_arr, count=num)

            inertia_arr = wp.array(inv_inertia_list, dtype=wp.mat33, device=self.model.device)
            wp.copy(dest=self.model.body_inv_inertia, src=inertia_arr, count=num)

            # 清零受力
            self.state_0.body_f.zero_()
            self.box_spawned = True

        for substep in range(self.sim_substeps):
            self.solver.step(self.state_0, self.state_1, None, None, self.sim_dt)

            qd_np = self.state_0.body_qd.numpy()
            qd_np[:, 0:3] *= 0.985
            qd_np[:, 3:6] *= 0.95
            self.state_0.body_qd.assign(qd_np)

            if self.box_spawned:
                if hasattr(self.solver, 'last_step_forces') and self.solver.last_step_forces is not None:
                    forces_wp = self.solver.last_step_forces
                    current_f_np = forces_wp.numpy()

                    if self.smoothed_body_f.shape != current_f_np.shape:
                        self.smoothed_body_f = np.zeros_like(current_f_np)

                    self.smoothed_body_f = (1.0 - self.filter_alpha) * self.smoothed_body_f + \
                                           self.filter_alpha * current_f_np

                    final_f_wp = wp.array(self.smoothed_body_f, dtype=wp.spatial_vector, device=self.model.device)
                    self.state_0.body_f.assign(final_f_wp)
                else:
                    self.state_0.body_f.zero_()

            # XPBD 刚体求解
            contacts = self.model.collide(self.state_0)
            self.solver_rb.step(self.state_0, self.state_0, None, contacts, self.sim_dt)

            # 状态同步 (Ping-pong buffer)
            wp.copy(self.state_1.body_q, self.state_0.body_q)
            wp.copy(self.state_1.body_qd, self.state_0.body_qd)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

    def test_final(self):
        voxel_size = self.solver.mpm_model.voxel_size
        newton.examples.test_particle_state(
            self.state_0,
            "all particles are above the ground",
            lambda q, qd: q[2] > -voxel_size,
        )

        if self.collider == "cube":
            cube_extents = wp.vec3(0.5, 2.0, 0.6) * 0.9
            cube_center = wp.vec3(0.75, 0, 0.9)
            cube_lower = cube_center - cube_extents
            cube_upper = cube_center + cube_extents
            newton.examples.test_particle_state(
                self.state_0,
                "all particles are outside the cube",
                lambda q, qd: not newton.utils.vec_inside_limits(q, cube_lower, cube_upper),
            )

        # Test that some particles are still high-enough
        if self.collider in ("concave", "cube"):
            max_z = np.max(self.state_0.particle_q.numpy()[:, 2])
            assert max_z > 0.8, "All particles have collapsed"

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)

        if self.show_normals:
            # for debugging purposes, we can visualize the collider normals
            _impulses, pos, _cid = self.solver.collect_collider_impulses(self.state_0)
            normals = self.state_0.collider_normal_field.dof_values

            normal_vecs = 0.25 * self.solver.mpm_model.voxel_size * normals
            root = pos
            mid = pos + normal_vecs
            tip = mid + normal_vecs

            # draw two segments per normal so we can visualize direction (red roots, orange tips)
            self.viewer.log_lines(
                "/normal_roots",
                starts=root,
                ends=mid,
                colors=wp.full(pos.shape[0], value=wp.vec3(0.8, 0.0, 0.0), dtype=wp.vec3),
            )
            self.viewer.log_lines(
                "/normal_tips",
                starts=mid,
                ends=tip,
                colors=wp.full(pos.shape[0], value=wp.vec3(1.0, 0.5, 0.3), dtype=wp.vec3),
            )
        else:
            self.viewer.log_lines("/normal_roots", None, None, None)
            self.viewer.log_lines("/normal_tips", None, None, None)

        self.viewer.end_frame()

    def render_ui(self, imgui):
        _changed, self.show_normals = imgui.checkbox("Show Normals", self.show_normals)

    @staticmethod
    def emit_particles(builder: newton.ModelBuilder, args):
        density = args.density
        voxel_size = args.voxel_size

        particles_per_cell = 3
        particle_lo = np.array(args.emit_lo)
        particle_hi = np.array(args.emit_hi)
        particle_res = np.array(
            np.ceil(particles_per_cell * (particle_hi - particle_lo) / voxel_size),
            dtype=int,
        )

        cell_size = (particle_hi - particle_lo) / particle_res
        cell_volume = np.prod(cell_size)

        radius = np.max(cell_size) * 0.5
        mass = np.prod(cell_volume) * density

        builder.add_particle_grid(
            pos=wp.vec3(particle_lo),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0),
            dim_x=particle_res[0] + 1,
            dim_y=particle_res[1] + 1,
            dim_z=particle_res[2] + 1,
            cell_x=cell_size[0],
            cell_y=cell_size[1],
            cell_z=cell_size[2],
            mass=mass,
            jitter=2.0 * radius,
            radius_mean=radius,
        )


if __name__ == "__main__":
    # Create parser that inherits common arguments and adds example-specific ones
    parser = newton.examples.create_parser()

    # Scene configuration
    parser.add_argument("--collider", default="cube", choices=["cube", "wedge", "concave", "none"], type=str)
    parser.add_argument("--emit-lo", type=float, nargs=3, default=[-1, -1, 2.0])
    parser.add_argument("--emit-hi", type=float, nargs=3, default=[1, 1, 3.5])
    parser.add_argument("--gravity", type=float, nargs=3, default=[0, 0, -10])
    parser.add_argument("--fps", type=float, default=60.0)
    parser.add_argument("--substeps", type=int, default=1)

    # Add MPM-specific arguments
    parser.add_argument("--density", type=float, default=1000.0)
    parser.add_argument("--air-drag", type=float, default=1.0)
    parser.add_argument("--critical-fraction", "-cf", type=float, default=0.0)

    parser.add_argument("--young-modulus", "-ym", type=float, default=1.0e6)
    parser.add_argument("--poisson-ratio", "-nu", type=float, default=0.3)
    parser.add_argument("--friction-coeff", "-mu", type=float, default=0.68)
    parser.add_argument("--damping", type=float, default=0.0)
    parser.add_argument("--yield-pressure", "-yp", type=float, default=1.0e12)
    parser.add_argument("--tensile-yield-ratio", "-tyr", type=float, default=0.0)
    parser.add_argument("--yield-stress", "-ys", type=float, default=0.0)
    parser.add_argument("--hardening", type=float, default=0.0)

    parser.add_argument("--grid-type", "-gt", type=str, default="sparse", choices=["sparse", "fixed", "dense"])
    parser.add_argument("--grid-padding", "-gp", type=int, default=0)
    parser.add_argument("--max-active-cell-count", "-mac", type=int, default=-1)
    parser.add_argument("--solver", "-s", type=str, default="gauss-seidel", choices=["gauss-seidel", "jacobi"])
    parser.add_argument("--transfer-scheme", "-ts", type=str, default="apic", choices=["apic", "pic"])

    parser.add_argument("--strain-basis", "-sb", type=str, default="P0", choices=["P0", "Q1"])
    parser.add_argument("--collider-basis", "-cb", type=str, default="Q1")

    parser.add_argument("--max-iterations", "-it", type=int, default=250)
    parser.add_argument("--tolerance", "-tol", type=float, default=1.0e-6)
    parser.add_argument("--voxel-size", "-dx", type=float, default=0.1)

    # Parse arguments and initialize viewer
    viewer, args = newton.examples.init(parser)

    # Create example and run
    example = Example(viewer, args)

    newton.examples.run(example, args)
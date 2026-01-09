import numpy as np
import warp as wp

import newton
import newton.examples
from newton.solvers import SolverImplicitMPM
from newton.solvers import SolverMLSMPM
from newton.solvers import SolverXPBD


class Example:
    def __init__(self, viewer, options):
        self.fps = options.fps
        self.frame_dt = 1.0 / self.fps

        self.sim_time = 0.0
        self.sim_substeps = options.substeps
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.viewer = viewer
        builder = newton.ModelBuilder()

        Example.emit_particles(builder, options)

        self.body_shape = options.body_shape
        self.rigid_bodies_config = [
            {'pos': (0.4, 0.4, 2.5), 'mass': 10.0, 'size': (0.35, 0.35, 0.35)}
        ]
        num_bodies = len(self.rigid_bodies_config)

        self.box_spawn_time = 1.0
        self.box_spawned = False
        self.filter_alpha = 0.15
        self.smoothed_body_f = np.zeros((num_bodies, 6), dtype=np.float32)

        hidden_pose = wp.transform(wp.vec3(999.0, 999.0, 999.0), wp.quat_identity())

        # 循环生成刚体
        for cfg in self.rigid_bodies_config:
            body_id = builder.add_body(
                mass=cfg['mass'],
                xform=hidden_pose
            )

            if self.body_shape == "sphere":
                radius = cfg['size'][0]
                builder.add_shape_sphere(
                    body=body_id,
                    cfg=newton.ModelBuilder.ShapeConfig(mu=0.5, density=0.0),
                    radius=radius,
                    xform=wp.transform_identity()
                )
            else:
                # 默认为 Box
                builder.add_shape_box(
                    body=body_id,
                    cfg=newton.ModelBuilder.ShapeConfig(mu=0.5, density=0.0),
                    hx=cfg['size'][0],
                    hy=cfg['size'][1],
                    hz=cfg['size'][2],
                    xform=wp.transform_identity()
                )

        # 地面
        builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=0.5))

        self.model = builder.finalize()
        self.model.particle_mu = options.friction_coeff
        self.model.set_gravity(options.gravity)

        self.model.particle_ke = 0.0
        self.model.particle_kd = 0.0
        self.model.particle_cohesion = 0.0
        self.model.particle_adhesion = 0.0

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

        # Initialize MPM solver
        self.solver = SolverMLSMPM(self.model, mpm_options)

        self.solver.enrich_state(self.state_0)
        self.solver.enrich_state(self.state_1)

        sizes_list = [wp.vec3(cfg['size']) for cfg in self.rigid_bodies_config]
        self.solver.body_sizes_buffer = wp.array(sizes_list, dtype=wp.vec3, device=self.model.device)

        self.solver.body_shape_type = 1 if self.body_shape == "sphere" else 0

        self.viewer.set_model(self.model)
        self.solver_rb = SolverXPBD(self.model)

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
        # 刚体生成逻辑
        if not self.box_spawned and self.sim_time >= self.box_spawn_time:
            print(f"--- Spawning {len(self.rigid_bodies_config)} {self.body_shape.title()}s ---")

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

                if self.body_shape == "sphere":
                    # Sphere: I = 2/5 * m * r^2
                    r = hx
                    i_val = 0.4 * m * r * r
                    inv_i_val = 1.0 / i_val if i_val > 1e-6 else 0.0
                    inv_inertia = wp.mat33(
                        inv_i_val, 0.0, 0.0,
                        0.0, inv_i_val, 0.0,
                        0.0, 0.0, inv_i_val
                    )
                else:
                    # Box: I = m/3 * (h^2 + h^2)
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

            # 拷贝数据到 GPU
            pos_arr = wp.array(new_pos_list, dtype=wp.transform, device=self.model.device)
            wp.copy(dest=self.state_0.body_q, src=pos_arr, count=num)
            wp.copy(dest=self.state_1.body_q, src=pos_arr, count=num)

            vel_arr = wp.array(new_vel_list, dtype=wp.spatial_vector, device=self.model.device)
            wp.copy(dest=self.state_0.body_qd, src=vel_arr, count=num)
            wp.copy(dest=self.state_1.body_qd, src=vel_arr, count=num)

            mass_arr = wp.array(inv_mass_list, dtype=float, device=self.model.device)
            wp.copy(dest=self.model.body_inv_mass, src=mass_arr, count=num)

            inertia_arr = wp.array(inv_inertia_list, dtype=wp.mat33, device=self.model.device)
            wp.copy(dest=self.model.body_inv_inertia, src=inertia_arr, count=num)

            self.state_0.body_f.zero_()
            self.box_spawned = True

        for substep in range(self.sim_substeps):
            self.solver.step(self.state_0, self.state_1, None, None, self.sim_dt)

            # 简单的阻尼
            qd_np = self.state_0.body_qd.numpy()
            qd_np[:, 0:3] *= 0.985
            qd_np[:, 3:6] *= 0.95
            self.state_0.body_qd.assign(qd_np)

            # 处理流固耦合力
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

            # 状态同步
            wp.copy(self.state_1.body_q, self.state_0.body_q)
            wp.copy(self.state_1.body_qd, self.state_0.body_qd)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def render_ui(self, imgui):
        _changed, self.show_normals = imgui.checkbox("Show Normals", self.show_normals)

    # --- 严格保留原始 granular 的粒子生成逻辑 ---
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
    parser = newton.examples.create_parser()

    # --- 添加新参数：形状选择 ---
    parser.add_argument("--body-shape", default="sphere", choices=["box", "sphere"], help="Shape of the rigid bodies")

    # --- 保留所有 granular 的参数设置 ---
    parser.add_argument("--collider", default="cube", choices=["cube", "wedge", "concave", "none"], type=str)
    parser.add_argument("--emit-lo", type=float, nargs=3, default=[-1, -1, 2.0])
    parser.add_argument("--emit-hi", type=float, nargs=3, default=[1, 1, 3.5])
    parser.add_argument("--gravity", type=float, nargs=3, default=[0, 0, -10])
    parser.add_argument("--fps", type=float, default=60.0)
    parser.add_argument("--substeps", type=int, default=1)  # 默认为1

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

    viewer, args = newton.examples.init(parser)

    example = Example(viewer, args)
    newton.examples.run(example, args)
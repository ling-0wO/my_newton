import numpy as np
import warp as wp
import sys
import os

import newton
import newton.examples
from newton.solvers import SolverImplicitMPM
from newton._src.solvers.mls_mpm.solver_mls_mpm import SolverMLSMPM, MLSMPMOptions
from newton._src.solvers.mls_mpm.mls_types import MLSMPMParticleData

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

        for key in vars(options):
            if hasattr(mpm_options, key):
                setattr(mpm_options, key, getattr(options, key))

        mpm_model = SolverImplicitMPM.Model(self.model, mpm_options)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()

        self.solver = SolverMLSMPM(self.model, mpm_options)

        self.solver.enrich_state(self.state_0)
        self.solver.enrich_state(self.state_1)

        self.viewer.set_model(self.model)
        self.viewer.show_particles = True

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
        for substep in range(self.sim_substeps):
            self.solver.step(self.state_0, self.state_1, None, None, self.sim_dt)
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

    parser.add_argument("--collider", default="cube", choices=["cube", "wedge", "concave", "none"], type=str)
    parser.add_argument("--emit-lo", type=float, nargs=3, default=[-1, -1, 2.0])
    parser.add_argument("--emit-hi", type=float, nargs=3, default=[1, 1, 3.5])
    parser.add_argument("--gravity", type=float, nargs=3, default=[0, 0, -10])
    parser.add_argument("--fps", type=float, default=60.0)
    parser.add_argument("--substeps", type=int, default=1)

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
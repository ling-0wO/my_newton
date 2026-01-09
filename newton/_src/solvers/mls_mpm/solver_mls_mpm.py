import warp as wp
import numpy as np
from ..implicit_mpm import SolverImplicitMPM
from .mls_types import MLSMPMParameters, MLSMPMParticleData
from .mls_mpm_core import MLSMPMCore
from . import mls_kernels as kernels

__all__ = ["SolverMLSMPM"]
class MLSMPMOptions(SolverImplicitMPM.Options):
    def __init__(self):
        super().__init__()
        self.support_radius = 0.15
        self.viscosity = 0.01
        self.fluid_density = 1000.0
        self.bulk_modulus = 1.0e6


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
        self.grid_contact_impulse = None

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
            padding = 0.4
            min_pos = np.min(positions, axis=0) - padding
            max_pos = np.max(positions, axis=0) + padding
            min_pos[2] = min(min_pos[2], -0.5)

            grid_dx = float(self.mls_core.params.grid_dx)
            if grid_dx <= 1e-4: grid_dx = 0.08

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
            self.grid_contact_impulse = wp.zeros(num_cells, dtype=wp.vec3, device=self.device)

            self._initialized = True
            return True

        except Exception as e:
            print(f"MLS Init Error: {e}")
            return False

    def _step_impl(self, state_in, state_out, dt, pic, scratch):
        if not self._initialized:
            if not self.initialize_mls_particles(state_in): return

        body_q = None
        body_qd = None
        if hasattr(state_in, 'body_q') and state_in.body_q is not None:
            body_q = state_in.body_q
        if hasattr(state_in, 'body_qd') and state_in.body_qd is not None:
            body_qd = state_in.body_qd

        if self.grid_contact_impulse is not None:
            self.grid_contact_impulse.zero_()
        # 自适应时间步长
        K = self.mls_core.params.bulk_modulus
        rho = self.mls_core.params.density
        dx = self.mls_core.params.grid_dx

        c_sound = np.sqrt(K / rho)
        v_max_est = 20.0

        cfl_alpha = 0.4
        max_dt = cfl_alpha * dx / (c_sound + v_max_est + 1e-6)

        num_sub_steps = int(np.ceil(dt / max_dt))
        num_sub_steps = max(1, min(num_sub_steps, 500))
        feedback_scale = 3.0 / float(num_sub_steps)
        sub_dt = dt / float(num_sub_steps)

        # if (state_in.step if hasattr(state_in, 'step') else 0) % 60 == 0:
        #    print(f"Step info: dt={dt:.4f} -> {num_sub_steps} substeps of {sub_dt:.2e}s. c={c_sound:.1f}")

        for _ in range(num_sub_steps):
            self._sub_step(sub_dt, body_q, body_qd, feedback_scale)

        if hasattr(state_out, 'particle_q'):
            wp.copy(state_out.particle_q, self.mls_particle_data.position)
        if hasattr(state_out, 'particle_qd'):
            wp.copy(state_out.particle_qd, self.mls_particle_data.velocity)

    def collect_collider_impulses(self, state):
        return self.grid_contact_impulse, None, None

    def _sub_step(self, dt, body_q=None, body_qd=None, feedback_scale=0.15):
        self.grid_mass.zero_()
        self.grid_mom.zero_()

        # 更新粒子应力 (Update Stress)
        wp.launch(
            kernels.update_particle_stress_kernel,
            dim=self.mls_particle_data.position.shape[0],
            inputs=[
                self.mls_particle_data,
                self.mls_core.params.viscosity,
                self.mls_core.params.bulk_modulus
            ],
            device=self.device
        )

        # 粒子到网格 (P2G)
        wp.launch(
            kernel=kernels.mls_p2g_apic_kernel,
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

        num_bodies = 0
        current_forces = None
        sizes_arr = None
        shape_type = 0
        if body_q is not None:
            num_bodies = body_q.shape[0]

            current_forces = wp.zeros(shape=(num_bodies,), dtype=wp.spatial_vector, device=self.device)

            if hasattr(self, 'body_sizes_buffer') and self.body_sizes_buffer is not None:
                if self.body_sizes_buffer.shape[0] == num_bodies:
                    sizes_arr = self.body_sizes_buffer
                else:
                    print(
                        f"Warning: body_sizes_buffer shape {self.body_sizes_buffer.shape[0]} != num_bodies {num_bodies}. Using default.")

            if sizes_arr is None:
                sizes_arr = wp.array([wp.vec3(0.35, 0.35, 0.35)] * num_bodies, dtype=wp.vec3, device=self.device)

        wp.launch(
            kernel=kernels.update_grid_with_coupling_kernel,
            dim=self.grid_mass.shape[0],
            inputs=[
                self.grid_mass,
                self.grid_mom,
                current_forces,
                self.mls_core.params.gravity,
                dt,
                self.mls_core.params.grid_dim,
                self.mls_core.params.grid_origin,
                self.mls_core.params.grid_dx,
                num_bodies,
                body_q,
                body_qd,
                sizes_arr,
                0.5,
                feedback_scale,
                shape_type
            ],
            device=self.device
        )

        self.last_step_forces = current_forces

        # 网格到粒子 (G2P)
        wp.launch(
            kernel=kernels.mls_g2p_apic_kernel,
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

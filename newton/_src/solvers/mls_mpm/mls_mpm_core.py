import warp as wp
from .mls_types import MLSMPMParameters, MLSMPMParticleData
import numpy as np

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
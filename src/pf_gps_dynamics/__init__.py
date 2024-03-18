from pf_gps_dynamics.__version__ import version
from sys import float_info
from typing import List, Tuple
import numpy as np
from numpy.random import uniform, randn
from tqdm import tqdm
from filterpy.monte_carlo import residual_resample


__all__ = ['version', 'create_uniform_particles', 'NetworkLearning']


def create_uniform_particles(ranges: List[Tuple[int, int]], n_stations: int,
                             n_particles: int, n_dimensions: int) -> np.ndarray:
    size = (n_particles, n_stations, n_stations, n_dimensions)
    particles = np.empty(size)
    for d in range(n_dimensions):
        low, high = ranges[d]
        samples = uniform(low=low, high=high, size=size[:3])
        particles[..., d] = samples
    return particles


class NetworkLearning:
    def __init__(self, observations: np.ndarray, interested_period: Tuple[int, int] = None,  # (20774, 21554),
                 n_particles: int = 1000, particle_noise_std: float = 0.01,
                 observation_noise_std: float = 0.01, prior_particles: np.ndarray = None,
                 laplace_obs_noise: bool = False, cauchy_obs_noise: bool = False):
        self.observations = observations
        self.n_stations = observations.shape[0]
        self.n_obs = observations.shape[1]
        self.n_dimensions = observations.shape[2]
        self.n_particles = n_particles

        self.laplace_obs_noise = laplace_obs_noise
        self.cauchy_obs_noise = cauchy_obs_noise

        self.particles = prior_particles if prior_particles is not None else self.initialize_particles()
        self.particle_noise_std = particle_noise_std
        self.observation_noise_std = observation_noise_std
        self.weights = np.ones((n_particles, self.n_stations, self.n_dimensions)) / n_particles
        self.interested_period = interested_period

        # initialize storage variables
        shape = (self.n_obs, self.n_stations, self.n_stations, self.n_dimensions)
        self.particle_means = np.empty(shape)
        self.particle_stds = np.empty(shape)
        self.all_particles_means = np.empty((self.n_obs, self.n_dimensions))
        self.all_particles_stds = np.empty((self.n_obs, self.n_dimensions))

        # self.learned_particles is of shape (self.interested_period[1] - self.interested_period[0] + 1, self.n_particles, self.n_stations, self.n_stations, self.n_dimensions)
        self.learned_particles = []

        # self.estimates is of shape (self.n_observations, self.n_stations, self.n_dimensions)
        self.estimates = []

    def initialize_particles(self) -> np.ndarray:
        # default prior particles
        dimension_value_ranges = [(-1, 1)] * self.n_dimensions
        return create_uniform_particles(dimension_value_ranges, self.n_stations, self.n_particles, self.n_dimensions)

    def predict(self, particles_prev_t: np.ndarray, observations_prev_t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

        particles_t = np.empty((self.n_particles, self.n_stations, self.n_stations, self.n_dimensions))
        for d in range(self.n_dimensions):
            particles_t[..., d] = particles_prev_t[..., d] \
                                          + (randn(self.n_particles, self.n_stations, self.n_stations)
                                             * self.particle_noise_std)

        predicted_observations_t = np.empty((self.n_particles, self.n_stations, self.n_dimensions))

        for d in range(self.n_dimensions):
            for station in range(self.n_stations):
                predicted_observations_t[:, station, d] = np.matmul(particles_t[:, station, :, d],
                                                                    observations_prev_t[:, d].T)

        return particles_t, predicted_observations_t

    def likelihood(self, observations_t: np.ndarray, predicted_obs_t: np.ndarray):
        """

        Args:
            observations_t:
            predicted_obs_t:

        Returns:
            a numpy array of shape (self.n_particles, self.n_stations, self.n_dimensions)

        """

        likelihood = np.empty(shape=(self.n_particles, self.n_stations, self.n_dimensions))

        for d in range(self.n_dimensions):
            for nth_station in range(self.n_stations):
                x = observations_t[nth_station, d]
                x_pred = predicted_obs_t[:, nth_station, d]
                if self.laplace_obs_noise:
                    beta = self.observation_noise_std / np.sqrt(2.0)
                    likelihood[:, nth_station, d] = np.exp(-np.abs(x - x_pred) / beta)  # ignore constant term
                elif self.cauchy_obs_noise:
                    gamma = self.observation_noise_std
                    likelihood[:, nth_station, d] = 1 / (
                        np.pi * gamma * (1 + ((x - x_pred) / gamma) ** 2)
                    )
                else:
                    # gaussian likelihood
                    # ignore constant term
                    likelihood[:, nth_station, d] = np.exp(-(x - x_pred) ** 2 / (
                                                                       2 * self.observation_noise_std ** 2))

        return likelihood

    def update(self, observations_t: np.ndarray,
               predicted_obs_t: np.ndarray):
        """

        Args:
            observations_t: with shape (self.n_stations, self.n_dimensions)
            predicted_obs_t: with shape (self.n_particles, self.n_stations, self.n_dimensions)

        Returns:
            a numpy array with shape (self.n_particles, self.n_stations, self.n_dimensions)
        """

        likelihood_t = self.likelihood(observations_t=observations_t, predicted_obs_t=predicted_obs_t)
        assert likelihood_t.shape == (self.n_particles, self.n_stations, self.n_dimensions)

        updated_weights = np.empty(shape=(self.n_particles, self.n_stations, self.n_dimensions))
        for d in range(self.n_dimensions):
            # EQ.11
            updated_weights[:, :, d] = likelihood_t[:, :, d]  # previous weights are uniform from resampling

        return updated_weights

    def learn(self):
        self.estimates = None
        for i in tqdm(range(self.n_obs-1)):
            t = i + 1
            update_particles, predicted_obs = self.predict(particles_prev_t=self.particles,
                                                           observations_prev_t=self.observations[:, t - 1, :])

            self.particles = update_particles

            new_particle_weights = self.update(observations_t=self.observations[:, t, :],
                                               predicted_obs_t=predicted_obs)

            self.weights = new_particle_weights
            for d in range(self.n_dimensions):
                for station in range(self.n_stations):
                    log_weights = np.log(self.weights[:, station, d] + float_info.epsilon)
                    max_log_weight = np.max(log_weights)
                    log_weights -= max_log_weight
                    self.weights[:, station, d] = np.exp(log_weights) / np.sum(np.exp(log_weights))  # normalization

            self.particles, self.weights = self.resampling(self.weights, self.particles)

            for d in range(self.n_dimensions):
                for station in range(self.n_stations):
                    for map_station in range(self.n_stations):
                        mean = np.average(self.particles[:, station, map_station, d])
                        std = np.std(self.particles[:, station, map_station, d])
                        self.particle_means[t, station, map_station, d] = mean
                        self.particle_stds[t, station, map_station, d] = std
                self.all_particles_means[t, d] = np.mean(self.particles[..., d])
                self.all_particles_stds[t, d] = np.std(self.particles[..., d])

            if self.interested_period is not None:
                if self.interested_period[0] <= t <= self.interested_period[1]:
                    self.write_learned_params(particles=self.particles.copy())

    def resampling(self, weights: np.ndarray, particles: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """

        Args:
            weights: with shape (self.n_particles, self.n_stations, self.n_dimensions)
            particles: with shape (self.n_particles, self.n_stations, self.n_stations, self.n_dimensions)

        Returns:
            a tuple of new particles and weights

        """

        new_weights = np.empty(shape=(self.n_particles, self.n_stations, self.n_dimensions))
        new_particles = np.empty(shape=(self.n_particles, self.n_stations, self.n_stations, self.n_dimensions))

        for d in range(self.n_dimensions):
            for station in range(self.n_stations):
                particle_indexes = residual_resample(weights[:, station, d])
                new_particles[:, station, :, d] = particles[particle_indexes, station, :, d]

                new_weights[:, station, d] = [1. / self.n_particles for _ in particle_indexes]

        return new_particles, new_weights

    def write_learned_params(self, particles: np.ndarray):
        self.learned_particles.append(particles)

    def get_estimates(self, recalculate=False) -> np.ndarray:
        """

        Args:
            recalculate:

        Returns:
            a numpy array of shape (self.n_stations, self.n_observations, self.n_dimensions)

        """
        if self.estimates is not None and not recalculate:
            return np.array(self.estimates)

        self.learned_particles = np.array(self.learned_particles)

        estimates = []

        for station in range(self.n_stations):
            all_stations_estimates = []
            for nth_obs in range(self.n_obs):
                nth_obs_estimates = []
                for d in range(self.n_dimensions):
                    hidden_params = self.particle_means[nth_obs, station, :, d]
                    station_value_estimate = np.matmul(hidden_params, self.observations[:, nth_obs - 1, d])
                    # ignore the estimate at first observation (nth_obs == 0), we dont predict the first observation.
                    nth_obs_estimates.append(station_value_estimate)
                all_stations_estimates.append(nth_obs_estimates)
            estimates.append(all_stations_estimates)
        self.estimates = np.array(estimates)

        return np.array(estimates)


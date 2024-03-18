# pf_gps_dynamics

This library is designed to apply particle filtering (PF) to analyze dynamics in GPS data, and accommodate different noise models for observations.

## installation:
```
# parent directory
cd pf_gps_dynamics
pip install -e .
```
## Parameters:

- `data`: A numpy array of shape `(n_stations, n_observations, n_dimensions)` representing the GPS data from n_stations, each with n_observations in n_dimensions (ex: north-south, east-west).
- `interested_period`: A tuple of time indices `(start, end)`, specifying the period of interest. Particles from `interested_period[0]` to `interested_period[1]` (inclusive) are retained.
- `prior_particles`: A numpy array of shape `(n_particles, n_stations, n_stations, n_dimensions)`. Default is uniform particles ranging from -1 to 1. Represents the initial distribution of particles for state estimation.
- `particle_noise_std`: The standard deviation of the noise term for state particles, indicating the uncertainty in particle transitions.
- `observation_noise_std`: The standard deviation of the noise term for observations, indicating the measurement noise.
- If both `laplace_obs_noise` and `cauchy_obs_noise` are `False`, Gaussian observation noise is assumed by default.
- If `laplace_obs_noise` or `cauchy_obs_noise` is `True`, it assumes Laplace or Cauchy observation noise, respectively.

## Usage:

```python
from pf_gps_dynamics import NetworkLearning

net = NetworkLearning(observations=data, interested_period=(start_idx, end_idx), n_particles=1000,
                      prior_particles=prior_particles, particle_noise_std=0.01, observation_noise_std=0.01,
                      laplace_obs_noise=False, cauchy_obs_noise=False
                     )
net.learn()

estimates = net.get_estimates()  # Predicted observations of shape (n_stations, n_observations, n_dimensions)
learned_particles = net.learned_particles  # Particles for hidden/state parameters of shape (interested_period_len, n_particles, n_stations, n_stations, n_dimensions)
particle_means = net.particle_means  # Estimated hidden/state parameters of shape (n_observations, n_stations, n_stations, n_dimensions)
particle_stds = net.particle_stds  # Standard deviations of estimated hidden/state parameters of shape (n_observations, n_stations, n_stations, n_dimensions)

# note: please ignore the first observation's estimates at index 0, we start the estimates at index 1.

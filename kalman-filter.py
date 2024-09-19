# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 12:13:19 2024

Kalman filter is a statistical technique to infer the currect state of a state space model by previous observations. 
This can be used to model an 'online prior' that is updated on a trial-by-trial basis
The kf continuously updates the estimate by combining the prior state estimate (from the previous step) and the new observation

To do:
    1) add condition 1
@author: YSK
"""
import numpy as np
import matplotlib.pyplot as plt

# Set initial values
initial_value = 52
final_value = 26
n_steps = 26
n_subjects = 50
condition = "condition_3"

def kf(start, end, n, s, figname=False, verbos=False, plot=False):
    
    # Kalman filter initialization
    state_transition = 1.0  # No change in state without external input
    observation_model = 1.0  # We observe the true value without any distortion
    process_noise = 0.5  # Process noise (how much we believe the state evolves unpredictably)
    
    states = np.linspace(start, end, n)  # True states
    observation_noise = 20.0  # Observation noise (uncertainty in measurements)
    observations = np.zeros(n)
    
    initial_state_estimate = start  # Initial guess of the state
    initial_covariance_estimate = 1.0  # Initial estimate of state uncertainty
    
    # to store results
    state_estimates = np.zeros(n)  # Store estimated states
    covariance_estimates = np.zeros(n)  # Store covariance estimates
    
    state_estimates[0] = initial_state_estimate
    covariance_estimates[0] = initial_covariance_estimate
    
    ##### which distribution to use to sample observations? ####
    initial_obs_sample = states[0] + np.random.normal(0, observation_noise, size=s)
    observations[0] = np.average(initial_obs_sample)  # Use the median of the samples

    # Kalman filter loop
    for t in range(1, n):
        ##### which distribution to use to sample observations? ####
        sample_observations = states[t] + np.random.normal(0, observation_noise, size=s)
        observations[t] = np.average(sample_observations)  # Use the average of the samples
        
        # Prediction step
        predicted_state = state_transition * state_estimates[t - 1]
        predicted_covariance = state_transition * covariance_estimates[t - 1] * state_transition + process_noise
    
        # Update step (using Bayesian approach)
        kalman_gain = predicted_covariance * observation_model / (observation_model * predicted_covariance * observation_model + observation_noise/s)
        state_estimates[t] = predicted_state + kalman_gain * (observations[t] - observation_model * predicted_state)
        covariance_estimates[t] = (1 - kalman_gain * observation_model) * predicted_covariance

        if verbos:
            print(f"-step: {t}, initial state: {predicted_state:.1f}, prediction: {state_estimates[t]:.1f}")
    
    print(f"process final prediction: {state_estimates[-1]:.1f}")
    
    if plot:
        # Plotting the results
        plt.plot(states, label='True State', linestyle='--', c='gray')
        obs_str = 'Observations (mean of N = ' + str(s) + ')'
        plt.plot(observations, label=obs_str, marker='o', c='darkcyan')
        plt.plot(state_estimates, label='Kalman Filter Estimate', marker='x',c='paleturquoise')
        plt.xlabel('Time Step')
        plt.ylabel('State Value [dots]')
        plt.legend()
        
    if figname:
        plt.title('Kalman Filter Bayesian Simulation for '+figname)
        plt.savefig(figname+'.jpeg', dpi=500)
        print("-figure saved!")
    else:
        plt.show()
    
kf(initial_value, final_value, n_steps, n_subjects, verbos=True, plot=True, figname=condition)




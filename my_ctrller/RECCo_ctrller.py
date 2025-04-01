import numpy as np
from collections import namedtuple
from typing import List, Dict, Tuple

# Define the Action namedtuple similar to the original controller
Action = namedtuple('Action', ['basal', 'bolus'])

class RECCoController:
    def __init__(self, bg_target=100.0, sample_time=1.0):
        """
        Initialize the RECCo controller with default parameters
        
        Parameters:
        - bg_target: Target blood glucose level (mg/dl), default 100 mg/dl
        - sample_time: Sampling time in minutes, default 1 minute
        """
        # Target blood glucose level
        self.bg_target = bg_target
        
        # Sampling time
        self.sample_time = sample_time
        
        # Controller parameters from Table 7 in the paper
        self.u_min = 0       # Minimum insulin dose (mU/L)
        self.u_max = 100     # Maximum insulin dose (mU/L)
        self.y_min = 20      # Minimum BG value (mg/dl)
        self.y_max = 70      # Maximum BG value (mg/dl)
        self.time_constant = 40  # Time constant for reference model
        
        # Evolving parameters
        self.gamma_max = 0.93
        self.k_add = 1
        self.n_add = 20
        self.C = 0           # Current number of clouds
        self.C_max = 20      # Maximum number of clouds
        
        # Adaptation parameters
        self.G_sign = 1      # Sign of process gain
        self.alpha_p = 0.1   # Proportional adaptation gain
        self.alpha_i = 0.1   # Integral adaptation gain
        self.alpha_d = 0.1   # Derivative adaptation gain
        self.alpha_r = 0.1   # Operating point adaptation gain
        
        # Data clouds storage
        self.clouds = []     # List to store cloud parameters
        self.data_points = [] # List to store historical data points
        
        # Tracking error variables
        self.error_integral = 0
        self.prev_error = 0
        
        # PID parameters for each cloud
        self.P = []          # Proportional gains
        self.I = []          # Integral gains
        self.D = []          # Derivative gains
        self.R = []          # Operating point compensations
        
    def policy(self, observation, reward, done, **info):
        """
        Main control policy that calculates insulin dose based on current BG
        
        Parameters:
        - observation: Current BG measurement
        - reward: Current reward (not used here)
        - done: Flag indicating episode end (not used here)
        - info: Additional information (patient_name, sample_time)
        
        Returns:
        - action: Namedtuple containing basal and bolus insulin doses
        """
        # Get current BG measurement
        bg = observation.CGM
        
        # Calculate tracking error
        error = self.bg_target - bg
        
        # Update integral and derivative terms
        error_derivative = error - self.prev_error
        self.error_integral += error
        
        # Normalize inputs for evolving law
        delta_y = self.y_max - self.y_min
        delta_e = delta_y / 2
        normalized_error = error / delta_e
        normalized_bg = (bg - self.y_min) / delta_y
        
        # Create current data point
        current_point = np.array([normalized_error, normalized_bg])
        
        # Update data clouds
        self._update_clouds(current_point)
        
        # Calculate control signal
        insulin_dose = self._calculate_control_signal(error, error_derivative)
        
        # Store previous error
        self.prev_error = error
        
        # Return action (only basal for now, bolus=0)
        return Action(basal=insulin_dose, bolus=0)
    
    def _update_clouds(self, current_point):
        """
        Update the data clouds based on the current data point
        
        Parameters:
        - current_point: Current normalized data point [error, bg]
        """
        if self.C == 0:
            # Initialize first cloud
            self._add_cloud(current_point)
        else:
            # Calculate membership values for existing clouds
            gamma_values = []
            for i in range(self.C):
                cloud = self.clouds[i]
                mu_i = cloud['mu']
                sigma_i = cloud['sigma']
                
                # Calculate local density (equation from Table 5)
                numerator = 1
                denominator = 1 + np.linalg.norm(current_point - mu_i)**2 + sigma_i - np.linalg.norm(mu_i)**2
                gamma_i = numerator / denominator
                gamma_values.append(gamma_i)
            
            # Check if we need to add a new cloud
            max_gamma = max(gamma_values) if gamma_values else 0
            if max_gamma < self.gamma_max and len(self.data_points) > (self.k_add + self.n_add):
                self._add_cloud(current_point)
            else:
                # Associate with closest cloud
                closest_idx = np.argmax(gamma_values)
                self._update_cloud_parameters(closest_idx, current_point)
    
    def _add_cloud(self, point):
        """
        Add a new data cloud
        
        Parameters:
        - point: Data point to initialize the new cloud
        """
        if self.C >= self.C_max:
            return  # Don't exceed maximum number of clouds
            
        new_cloud = {
            'mu': point,                   # Mean
            'sigma': np.linalg.norm(point)**2,  # Variance
            'M': 1                         # Number of points in cloud
        }
        
        self.clouds.append(new_cloud)
        self.C += 1
        
        # Initialize PID parameters for this cloud
        self.P.append(0.5)  # Initial proportional gain
        self.I.append(0.1)  # Initial integral gain
        self.D.append(0.1)  # Initial derivative gain
        self.R.append(0)    # Initial operating point compensation
    
    def _update_cloud_parameters(self, cloud_idx, point):
        """
        Update parameters of an existing cloud
        
        Parameters:
        - cloud_idx: Index of the cloud to update
        - point: New data point to incorporate
        """
        cloud = self.clouds[cloud_idx]
        M = cloud['M']
        
        # Update mean (equation from Table 5)
        new_mu = ((M - 1) / M) * cloud['mu'] + (1 / M) * point
        cloud['mu'] = new_mu
        
        # Update variance (equation from Table 5)
        new_sigma = ((M - 1) / M) * cloud['sigma'] + (1 / M) * np.linalg.norm(point)**2
        cloud['sigma'] = new_sigma
        
        # Update count
        cloud['M'] += 1
    
    def _calculate_control_signal(self, error, error_derivative):
        """
        Calculate the control signal (insulin dose) using the RECCo algorithm
        
        Parameters:
        - error: Current tracking error
        - error_derivative: Derivative of tracking error
        
        Returns:
        - insulin_dose: Calculated insulin dose
        """
        if self.C == 0:
            return 0  # No clouds yet, return zero dose
            
        # Calculate relative densities for all clouds
        gamma_values = []
        for i in range(self.C):
            cloud = self.clouds[i]
            mu_i = cloud['mu']
            sigma_i = cloud['sigma']
            
            numerator = 1
            denominator = 1 + np.linalg.norm(np.array([error, self.prev_error]) - mu_i)**2 + sigma_i - np.linalg.norm(mu_i)**2
            gamma_i = numerator / denominator
            gamma_values.append(gamma_i)
        
        total_gamma = sum(gamma_values)
        lambda_values = [gamma / total_gamma for gamma in gamma_values]
        
        # Calculate local control signals for each cloud
        u_local = []
        for i in range(self.C):
            # PID-R control law (equation 34)
            u_i = (self.P[i] * error + 
                   self.I[i] * self.error_integral + 
                   self.D[i] * error_derivative + 
                   self.R[i])
            u_local.append(u_i)
        
        # Calculate weighted average control signal
        insulin_dose = sum(lambda_values[i] * u_local[i] for i in range(self.C))
        
        # Apply saturation
        insulin_dose = np.clip(insulin_dose, self.u_min, self.u_max)
        
        # Update adaptation parameters
        self._update_adaptation_parameters(error, error_derivative, lambda_values)
        
        return insulin_dose
    
    def _update_adaptation_parameters(self, error, error_derivative, lambda_values):
        """
        Update the adaptation parameters for each cloud
        
        Parameters:
        - error: Current tracking error
        - error_derivative: Derivative of tracking error
        - lambda_values: Relative densities for each cloud
        """
        for i in range(self.C):
            # Calculate adaptation terms (equations 37)
            delta_P = (self.alpha_p * self.G_sign * lambda_values[i] * 
                      abs(error * self.error_integral) / (1 + error**2))
            delta_I = (self.alpha_i * self.G_sign * lambda_values[i] * 
                      abs(error * error_derivative) / (1 + error**2))
            delta_D = (self.alpha_d * self.G_sign * lambda_values[i] * 
                      abs(error * error_derivative) / (1 + error**2))
            delta_R = (self.alpha_r * self.G_sign * lambda_values[i] * 
                      abs(error) / (1 + error**2))
            
            # Apply parameter projection (from Table 6)
            self.P[i] = np.clip(self.P[i] + delta_P, 0, None)
            self.I[i] = np.clip(self.I[i] + delta_I, 0, None)
            self.D[i] = np.clip(self.D[i] + delta_D, 0, None)
            self.R[i] += delta_R
    
    def reset(self):
        """
        Reset the controller to initial state
        """
        self.clouds = []
        self.data_points = []
        self.C = 0
        self.P = []
        self.I = []
        self.D = []
        self.R = []
        self.error_integral = 0
        self.prev_error = 0

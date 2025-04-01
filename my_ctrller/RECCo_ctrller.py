from .base import Controller
from .base import Action
import numpy as np
import pandas as pd
import pkg_resources
import logging
from collections import deque

logger = logging.getLogger(__name__)
CONTROL_QUEST = pkg_resources.resource_filename('simglucose',
                                                'params/Quest.csv')
PATIENT_PARA_FILE = pkg_resources.resource_filename(
    'simglucose', 'params/vpatient_params.csv')


class RECCoController(Controller):
    """
    Robust Evolving Cloud-based Controller for Type 1 Diabetes Management.
    Implements the ANYA fuzzy rule-based system as described in the research paper.
    """

    def __init__(self, target=100):
        self.quest = pd.read_csv(CONTROL_QUEST)
        self.patient_params = pd.read_csv(PATIENT_PARA_FILE)
        self.target = target  # Target blood glucose level (100 mg/dl)

        # Initialize RECCo parameters
        self.u_min = 0
        self.u_max = 100
        self.y_min = 20
        self.y_max = 70
        self.Ts = 2  # Sampling time (minutes)
        self.tau = 40  # Time constant

        # Evolving parameters
        self.gamma_max = 0.93
        self.k_add = 1
        self.n_add = 20
        self.C = 0  # Current number of clouds
        self.C_max = 20  # Maximum number of clouds

        # Adaptation parameters
        self.G_sign = 1
        self.alpha_p = 0.1
        self.alpha_i = 0.1
        self.alpha_d = 0.1
        self.alpha_r = 0.1

        # Data storage for evolving clouds
        self.clouds = []
        self.error_history = deque(maxlen=10)
        self.control_history = deque(maxlen=10)

        # Initialize first cloud when first data arrives
        self.initialized = False

    def policy(self, observation, reward, done, **kwargs):
        sample_time = kwargs.get('sample_time', 1)
        pname = kwargs.get('patient_name')
        meal = kwargs.get('meal')  # unit: g/min
        exercise = kwargs.get('exercise', 0)  # Exercise factor (0-1)
        stress = kwargs.get('stress', 0)  # Stress factor (0-1)
        fatigue = kwargs.get('fatigue', 0)  # Fatigue factor (0-1)

        # Get current glucose level
        current_glucose = observation.CGM

        # Calculate control action using RECCo algorithm
        action = self._recco_policy(pname, current_glucose, meal, exercise,
                                    stress, fatigue, sample_time)

        return action

    def _recco_policy(self, name, glucose, meal, exercise, stress, fatigue, env_sample_time):
        """
        RECCo control algorithm implementation based on the paper.
        """
        # 1. Reference Model
        y_r = self._reference_model(glucose)

        # 2. Calculate tracking error
        error = y_r - glucose
        delta_error = error - (self.error_history[-1] if len(self.error_history) > 0 else 0)

        # Store current error
        self.error_history.append(error)

        # 3. Normalize inputs for evolving system
        x = np.array([
            error / ((self.y_max - self.y_min) / 2),
            (glucose - self.y_min) / (self.y_max - self.y_min)
        ])

        # 4. Evolving Law - Cloud management
        if not self.initialized:
            # Initialize first cloud
            self._add_cloud(x)
            self.initialized = True
        else:
            # Calculate membership to existing clouds
            gamma_values = [self._calculate_local_density(x, cloud) for cloud in self.clouds]
            max_gamma = max(gamma_values) if self.clouds else 0

            # Add new cloud if conditions met
            if max_gamma < self.gamma_max and len(self.error_history) > self.n_add:
                self._add_cloud(x)

        # 5. Adaptation Law - PID-R type controller
        u_total = 0
        sum_gamma = 0

        for i, cloud in enumerate(self.clouds):
            # Calculate local density for this cloud
            gamma = self._calculate_local_density(x, cloud)

            # Calculate normalized relative density
            lambda_i = gamma / sum([self._calculate_local_density(x, c) for c in self.clouds])

            # Calculate PID-R components for this cloud
            P = cloud['P'] * error
            I = cloud['I'] * sum(self.error_history)
            D = cloud['D'] * delta_error
            R = cloud['R']

            # Calculate local control action
            u_i = P + I + D + R

            # Update adaptation parameters
            delta_P = self.alpha_p * self.G_sign * lambda_i * abs(error * error) / (1 + error ** 2)
            delta_I = self.alpha_i * self.G_sign * lambda_i * abs(error * delta_error) / (1 + error ** 2)
            delta_D = self.alpha_d * self.G_sign * lambda_i * abs(error * delta_error) / (1 + error ** 2)
            delta_R = self.alpha_r * self.G_sign * lambda_i * abs(error) / (1 + error ** 2)

            # Update cloud parameters with instability protection
            cloud['P'] = self._parameter_projection(cloud['P'] + delta_P)
            cloud['I'] = self._parameter_projection(cloud['I'] + delta_I)
            cloud['D'] = self._parameter_projection(cloud['D'] + delta_D)
            cloud['R'] = self._parameter_projection(cloud['R'] + delta_R)

            # Add to total control action
            u_total += lambda_i * u_i
            sum_gamma += gamma

        # Apply saturation to control action
        u_total = np.clip(u_total, self.u_min, self.u_max)

        # Convert to insulin rate (U/min)
        insulin_rate = u_total / env_sample_time

        # Apply meal bolus if needed (combining RECCo with meal response)
        if meal > 0:
            if any(self.quest.Name.str.match(name)):
                quest = self.quest[self.quest.Name.str.match(name)]
                bolus = (meal * env_sample_time) / quest.CR.values.item()
            else:
                bolus = (meal * env_sample_time) / 15  # Default CR

            bolus = bolus / env_sample_time  # Convert to U/min
            insulin_rate += bolus

        # Adjust for exercise, stress, fatigue factors
        insulin_rate *= (1 - 0.3 * exercise)  # Reduce insulin during exercise
        insulin_rate *= (1 + 0.2 * stress)  # Increase insulin during stress
        insulin_rate *= (1 - 0.1 * fatigue)  # Reduce insulin when fatigued

        # Store control action
        self.control_history.append(insulin_rate)

        return Action(basal=insulin_rate, bolus=0)  # RECCo handles both as single control

    def _reference_model(self, y):
        """
        First-order linear reference model.
        """
        a_r1 = 1 - self.Ts / self.tau
        return a_r1 * y + (1 - a_r1) * self.target

    def _add_cloud(self, x):
        """
        Add a new data cloud to the evolving system.
        """
        if self.C >= self.C_max:
            # Remove oldest cloud if at maximum
            self.clouds.pop(0)

        new_cloud = {
            'mean': x,
            'sigma': np.linalg.norm(x) ** 2,
            'P': 0.5,  # Initial proportional gain
            'I': 0.1,  # Initial integral gain
            'D': 0.05,  # Initial derivative gain
            'R': 0  # Initial operating point compensation
        }
        self.clouds.append(new_cloud)
        self.C += 1

    def _calculate_local_density(self, x, cloud):
        """
        Calculate local density of point x relative to a cloud.
        """
        return 1 / (1 + np.linalg.norm(x - cloud['mean']) ** 2 + cloud['sigma'] - np.linalg.norm(cloud['mean']) ** 2)

    def _parameter_projection(self, param):
        """
        Project parameter to valid range to prevent instability.
        """
        # Simple projection - can be enhanced based on Table 6 in the paper
        return np.clip(param, -10, 10)

    def reset(self):
        """Reset the controller state"""
        self.clouds = []
        self.error_history.clear()
        self.control_history.clear()
        self.C = 0
        self.initialized = False
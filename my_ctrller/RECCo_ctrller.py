import numpy as np
from collections import namedtuple
from .base import Controller, Action


class DataCloud:
    def __init__(self, x, u):
        self.focal_point = x  # Focal point is the most representative data point
        self.mean = x  # Mean of the cloud
        self.sigma = np.ones_like(x)  # Local scatter
        self.radius = 1.0  # Approximate spread of the cloud
        self.M = 1  # Number of samples in the cloud
        self.Sigma = np.dot(x, x)  # Scalar product of data
        self.u = u  # Control action associated with the cloud
        self.P = 0.0  # Proportional gain for this cloud
        self.global_density = 1.0  # Global density of focal point
        self.local_density = 1.0  # Local density of focal point

    def update(self, x, u, rho=0.5):
        # Update mean recursively
        new_mean = (self.M / (self.M + 1)) * self.mean + (1 / (self.M + 1)) * x

        # Update Sigma recursively
        new_Sigma = (self.M / (self.M + 1)) * self.Sigma + (1 / (self.M + 1)) * np.dot(x, x)

        # Update local scatter
        new_sigma = np.sqrt((self.M / (self.M + 1)) * self.sigma ** 2 +
                            (1 / (self.M + 1)) * (x - self.focal_point) ** 2)

        # Update radius
        new_radius = rho * self.radius + (1 - rho) * np.mean(new_sigma)

        # Update counts
        self.M += 1
        self.mean = new_mean
        self.Sigma = new_Sigma
        self.sigma = new_sigma
        self.radius = new_radius

    def calculate_local_density(self, x):
        # Cauchy kernel for local density calculation
        distance_sq = np.sum((x - self.mean) ** 2)
        return 1 / (1 + distance_sq + self.Sigma - np.dot(self.mean, self.mean))

    def update_focal_point(self, x, global_density, local_density):
        # Update if new point is more representative
        if (local_density > self.local_density and
                global_density > self.global_density):
            self.focal_point = x
            self.local_density = local_density
            self.global_density = global_density
            return True
        return False


class RECCoController(Controller):
    def __init__(self, G_sign=1, a_r=0.925, gamma_p=0.1, dead_zone=0.5,
                 sigma_p=1e-5, u_min=-6, u_max=6):
        """
        Robust Evolving Cloud-based Controller (RECCo)

        Parameters:
        - G_sign: Sign of plant monotonicity (+1 or -1)
        - a_r: Reference model pole (0 < a_r < 1)
        - gamma_p: Adaptive gain for proportional controller
        - dead_zone: Dead zone threshold for adaptation
        - sigma_p: Leakage term for adaptive law
        - u_min, u_max: Actuator constraints
        """
        self.clouds = []  # List of DataCloud objects
        self.G_sign = G_sign
        self.a_r = a_r
        self.gamma_p = gamma_p
        self.dead_zone = dead_zone
        self.sigma_p = sigma_p
        self.u_min = u_min
        self.u_max = u_max

        # Tracking variables
        self.y_ref_prev = 0.0  # Previous reference model output
        self.error_integral = 0.0  # Integral of tracking error
        self.error_prev = 0.0  # Previous tracking error
        self.global_mean = None  # Global mean of all data
        self.global_Sigma = None  # Global scalar product

    def policy(self, observation, reward, done, **info):
        """
        RECCo control policy implementation

        Inputs:
        - observation: Contains blood glucose level (observation.CGM)
        - reward: Current reward from environment
        - done: Whether episode is done
        - info: Additional info (patient_name, sample_time)

        Output:
        - action: Controller action (basal, bolus)
        """
        # Extract glucose level from observation
        y_k = observation.CGM

        # For diabetes control, we can consider the reference as a target glucose level
        # (e.g., 120 mg/dL for non-diabetic range)
        r_k = 120.0  # Target glucose level

        # Reference model output (first-order filter)
        y_ref_k = self.a_r * self.y_ref_prev + (1 - self.a_r) * r_k
        self.y_ref_prev = y_ref_k

        # Tracking error
        error_k = y_k - y_ref_k

        # Integral and derivative of error (for PID control)
        error_integral = self.error_integral + error_k
        error_derivative = error_k - self.error_prev

        # Update previous error
        self.error_prev = error_k
        self.error_integral = error_integral

        # Controller input vector (reference and tracking error)
        x_k = np.array([r_k, error_k])

        # Calculate control action
        u_k = self._calculate_control(x_k, error_k, error_integral, error_derivative)

        # Apply actuator constraints
        u_k = np.clip(u_k, self.u_min, self.u_max)

        # Create action (convert to basal insulin, no bolus for this controller)
        # Note: May need scaling factor to convert control signal to insulin units
        basal = max(0, u_k / 10.0)  # Example scaling - adjust based on your system
        action = Action(basal=basal, bolus=0)

        return action

    def _calculate_control(self, x_k, error_k, error_integral, error_derivative):
        """
        Calculate control action using RECCo algorithm
        """
        # Initialize controller if empty
        if len(self.clouds) == 0:
            # Initial control action (could be zero or small value)
            u_k = 0.0

            # Create first cloud
            new_cloud = DataCloud(x_k, u_k)
            self.clouds.append(new_cloud)

            # Initialize global statistics
            self.global_mean = x_k
            self.global_Sigma = np.dot(x_k, x_k)

            return u_k

        # Calculate global density for current point (including control from previous step)
        z_k = np.append(x_k, self.clouds[-1].u)  # Last control action

        if self.global_mean is None:
            global_density = 1.0
        else:
            distance_sq = np.sum((z_k - self.global_mean) ** 2)
            global_density = 1 / (1 + distance_sq + self.global_Sigma -
                                  np.dot(self.global_mean, self.global_mean))

        # Update global statistics
        k = len(self.clouds) + 1  # Approximate count of all samples
        self.global_mean = ((k - 1) / k) * self.global_mean + (1 / k) * z_k
        self.global_Sigma = ((k - 1) / k) * self.global_Sigma + (1 / k) * np.dot(z_k, z_k)

        # Calculate membership degrees to each cloud
        local_densities = [cloud.calculate_local_density(x_k) for cloud in self.clouds]
        total_density = sum(local_densities)
        lambda_k = [ld / total_density for ld in local_densities] if total_density > 0 else [1.0 / len(
            self.clouds)] * len(self.clouds)

        # Calculate control action from each cloud (using proportional control)
        u_k_list = [cloud.P * error_k for cloud in self.clouds]

        # Weighted average of control actions
        u_k = sum(l * u for l, u in zip(lambda_k, u_k_list))

        # Check conditions for adding new cloud
        add_new_cloud = True
        for cloud in self.clouds:
            # Check if any existing cloud is too close
            distance = np.linalg.norm(x_k - cloud.focal_point)
            if distance <= cloud.radius / 2:
                add_new_cloud = False
                break

        # Also need global density higher than all existing clouds
        if add_new_cloud:
            for cloud in self.clouds:
                if global_density <= cloud.global_density:
                    add_new_cloud = False
                    break

        # Add new cloud if conditions are met
        if add_new_cloud:
            # Initialize P for new cloud to maintain smooth transition
            if len(self.clouds) > 0:
                # Calculate what P would give same output at focal point
                # as previous configuration
                prev_u = sum(c.calculate_local_density(x_k) * c.P * error_k
                             for c in self.clouds) / sum(c.calculate_local_density(x_k)
                                                         for c in self.clouds)
                new_P = prev_u / error_k if error_k != 0 else np.mean([c.P for c in self.clouds])
            else:
                new_P = 0.0

            # Create new cloud
            new_cloud = DataCloud(x_k, u_k)
            new_cloud.P = new_P
            new_cloud.global_density = global_density
            new_cloud.local_density = max(local_densities) if local_densities else 1.0

            # Initialize scatter as average of existing clouds
            if len(self.clouds) > 0:
                new_cloud.sigma = np.mean([c.sigma for c in self.clouds], axis=0)
                new_cloud.radius = np.mean([c.radius for c in self.clouds])

            self.clouds.append(new_cloud)

        # Update existing clouds
        for i, cloud in enumerate(self.clouds):
            # Update cloud statistics
            cloud.update(x_k, u_k)

            # Update focal point if current point is more representative
            cloud.update_focal_point(x_k, global_density, local_densities[i])

            # Adapt P parameter (with robust modifications)
            if abs(error_k) >= self.dead_zone and (self.u_min <= u_k <= self.u_max):
                delta_P = self.gamma_p * self.G_sign * lambda_k[i] * error_k

                # Apply leakage
                cloud.P = (1 - self.sigma_p) * cloud.P + delta_P

                # Apply projection (ensure P has same sign as G_sign)
                if self.G_sign > 0:
                    cloud.P = max(0, cloud.P)
                else:
                    cloud.P = min(0, cloud.P)

        return u_k

    def reset(self):
        """
        Reset the controller to initial state
        """
        self.clouds = []
        self.y_ref_prev = 0.0
        self.error_integral = 0.0
        self.error_prev = 0.0
        self.global_mean = None
        self.global_Sigma = None
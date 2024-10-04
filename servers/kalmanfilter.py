import numpy as np


class KF:

    def __init__(self, initial_x, initial_xv, initial_z, initial_z_v, dt=1.0):
        # mean of the state vector
        self.x = np.array([initial_x, initial_xv, initial_z, initial_z_v])

        self.dt = dt

        # covariance of state vector
        self.P = np.eye(4)

        # state transmition matrix (under the assumption, velocity is constant)
        self.F = np.array([
            [1.0, dt, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, dt],
            [0.0, 0.0, 0.0, 1.0]
        ])

        # Process noise covariance matrix
        self.Q = np.eye(4) * 0.1
        # Observation Matrix, observing positions and not velocities
        self.H = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0]
        ])
        # Measurement Noise Covariance
        self.R = np.eye(2) * 0.1

    def predict(self):
        '''
            this method only updates (calculates)
            the state prediction and the error covariance
        '''
        # state prediction
        self.x = self.F @ self.x
        # error covariance prediction
        # self.P = self.F @ self.P @ self.F.T + self.Q

        self.P = self.F @ (self.P @ self.F.T) + self.Q
        return self.x, self.P

    def estimate(self, measurement):
        '''
            this method calculates the estimate of the
            state vector, the Kalman Gain and updates
            the Error covariance.
        '''

        # True Measurement
        z = np.array([[measurement[0]], [measurement[1]]])
        # Kalman Innovation S = (H P Ht + R)′1
        # S = self.H @ self.P @ self.H.T + self.R
        S = self.H @ (self.P @ self.H.T) + self.R
        # Kalman Gain
        # K = self.P @ self.H.T @ np.linalg.inv(S)
        K = self.P @ (self.H.T @ np.linalg.inv(S))

        # Updating the state estimate
        # Measurement Residual
        y = z - (self.H @ self.x.reshape(-1, 1))
        # State Estimate
        self.x = self.x.reshape(-1, 1) + (K @ y)

        # Updating Error Covariance
        # Identity matrix with the same size as the state vector
        identity = np.eye(self.F.shape[0])
        self.P = (identity - K @ self.H) @ self.P

        return self.x


class EKF:
    def __init__(
        self,
        initial_x,
        initial_z,
        initial_vx,
        initial_vz,
        initial_theta,
        initial_omega,
        dt
    ):
        self.dt = dt
        self.x = np.array(
            [initial_x,
             initial_z,
             initial_vx,
             initial_vz,
             initial_theta,
             initial_omega]
        )
        self.P = np.eye(6) * 1000  # High initial uncertainty

        # Process noise covariance
        self.Q = np.diag([0.1, 0.1, 1, 1, 0.1, 0.1])

        # Measurement noise covariance
        self.R = np.diag([0.1, 0.1, 0.5, 0.5, 1, 1])

    def predict(self):
        # State transition function
        x, z, vx, vz, theta, omega = self.x
        self.x = np.array([
            x + vx * self.dt,
            z + vz * self.dt,
            vx,
            vz,
            theta + omega * self.dt,
            omega
        ])

        # Jacobian of state transition function
        F = np.array([
            [1, 0, self.dt, 0, 0, 0],
            [0, 1, 0, self.dt, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, self.dt],
            [0, 0, 0, 0, 0, 1]
        ])

        self.P = F @ self.P @ F.T + self.Q
        return self.x

    def update(self, measurement):
        # Measurement function
        H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0]
        ])

        y = measurement - H @ self.x
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ H) @ self.P

    def get_state(self):
        return self.x

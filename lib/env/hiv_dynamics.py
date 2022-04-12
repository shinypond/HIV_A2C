import numpy as np
from scipy.integrate import solve_ivp
import torch

from .base import BaseEnv


class HIV(BaseEnv):
    # Dynamics hyperparameters
    lmbd1 = 10000
    lmbd2 = 31.98
    d1 = 0.01
    d2 = 0.01
    k1 = 8.0e-7
    k2 = 1.0e-4
    f = 0.34
    delta = 0.7
    m1 = 1.0e-5
    m2 = 1.0e-5
    N_T = 100
    c = 13
    rho1 = 1
    rho2 = 1
    lmbd_E = 1
    b_E = 0.3
    d_E = 0.25
    K_b = 100
    K_d = 500
    delta_E = 0.1

    # Action hyperparameters
    min_a1 = 0.0
    max_a1 = 0.7
    min_a2 = 0.0
    max_a2 = 0.3

    # Reward hyperparameters
    scaler = 1.0e+6
    Q = 0.1
    R1 = 20000
    R2 = 20000
    S = 1000

    def __init__(self, config, batch_size=None):
        self.init_state = np.array(config.env.init_state)
        if batch_size == None:
            self.batch_size = config.env.batch_size
        else:
            self.batch_size = batch_size
        self.T_max = config.env.T_max
        self.method = config.env.method
        self.reset()

    def reset(self):
        self.state = self.init_state.reshape(1, -1)
        self.state = np.repeat(self.state, self.batch_size, axis=0) 
        self.t_now = 0

    def step(self, action: np.ndarray):
        B = self.batch_size
        reward = np.zeros((B, 1))
        assert action.shape == (B, 2)
        x = np.concatenate([self.state, action, reward], axis=-1)
        x = x.reshape(-1)

        def ode_ftn(t, x):
            x = x.reshape(B, -1)
            dx = np.zeros_like(x)
            dx[:, 0] = self.lmbd1 - self.d1 * x[:, 0] - (1 - x[:, 6]) * self.k1 * x[:, 4] * x[:, 0]
            dx[:, 1] = self.lmbd2 - self.d2 * x[:, 1] - (1 - self.f * x[:, 6]) * self.k2 * x[:, 4] * x[:, 1]
            dx[:, 2] = (1 - x[:, 6]) * self.k1 * x[:, 4] * x[:, 0] - self.delta * x[:, 2] - self.m1 * x[:, 5] * x[:, 2]
            dx[:, 3] = (1 - self.f * x[:, 6]) * self.k2 * x[:, 4] * x[:, 1] - self.delta * x[:, 3] - self.m2 * x[:, 5] * x[:, 3]
            dx[:, 4] = (1 - x[:, 7]) * self.N_T * self.delta * (x[:, 2] + x[:, 3]) - self.c * x[:, 4] - \
                ((1 - x[:, 6]) * self.rho1 * self.k1 * x[:, 0] + (1 - self.f * x[:, 6]) * self.rho2 * self.k2 * x[:, 1]) * x[:, 4]
            _I = x[:, 2] + x[:, 3]
            _E_first = self.b_E * _I / (_I + self.K_b + 1e-16) * x[:, 5]
            _E_second = self.d_E * _I / (_I + self.K_d + 1e-16) * x[:, 5]
            dx[:, 5] = self.lmbd_E + _E_first - _E_second - self.delta_E * x[:, 5]
            dx[:, 8] = -(self.Q * x[:, 4] + self.R1 * x[:, 6] ** 2 + self.R2 * x[:, 7] ** 2 - self.S * x[:, 5])
            dx = dx.reshape(-1)
            return dx

        sol = solve_ivp(ode_ftn, (0, 1), x, rtol=1e-5, atol=1e-5, method=self.method)

        y = sol.y[:, -1].reshape(B, -1)
        self.state = y[:, :6] # Next state (observation)
        reward = y[:, -1] / self.scaler

        self.t_now += 1
        done = 0 if self.t_now < self.T_max else 1

        return self.state, reward, done

    def trim_action(self, action: np.ndarray):
        action[:, 0] = np.clip(action[:, 0], self.min_a1, self.max_a1)
        action[:, 1] = np.clip(action[:, 1], self.min_a2, self.max_a2)
        return action
        

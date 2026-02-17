import numpy as np

class OrnsteinUhlenbeckNoise:
    """
    实现了 OU 噪声，产生具有时间相关性的随机过程，
    能有效防止电流动作在相邻步长间发生剧烈跳变。
    """
    def __init__(self, mu=0.0, theta=0.15, sigma=0.2, dt=1e-2):
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.dt = dt
        self.reset()

    def reset(self):
        self.state = np.ones(1) * self.mu

    def sample(self):
        dx = self.theta * (self.mu - self.state) * self.dt + \
             self.sigma * np.sqrt(self.dt) * np.random.randn(1)
        self.state += dx
        return self.state
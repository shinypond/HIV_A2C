class BaseEnv:
    def __init__(self):
        self.state = None

    def reset(self):
        raise NotImplementedError

    def step(self, action):
        raise NotImplementedError
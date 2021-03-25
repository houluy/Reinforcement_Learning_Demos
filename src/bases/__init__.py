import random

class Observation:
    def __init__(self, space=None):
        self.space = space
        self.dimension = self.space.shape()[0]


class Action:
    def __init__(self, dimension=1):
        self.dimension = dimension

    def sample():
        pass



class DiscreteAction(Action):
    def __init__(self, dimension=1):
        super().__init__(dimension)
        self.actions = actions
        self.size = len(self.actions)

    def sample(self):
        random.choice(self.actions)


class ContinuousAction(Action):
    def __init__(self, dimension=1):
        super().__init__(dimension)


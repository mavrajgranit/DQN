import torch

class Stack:
    stack = []

    def __init__(self,stack_size,preprocessor):
        self.stack_size=stack_size
        self.preprocessor = preprocessor

    def reset(self,observation):
        self.stack = []
        obs = self.preprocessor.preprocess(observation)
        for i in range(self.stack_size):
            self.stack.append(obs)
        return torch.cat(self.stack,1)

    def step(self,observation):
        self.stack.pop(0)
        self.stack.append(self.preprocessor.preprocess(observation))
        return torch.cat(self.stack,1)

class StackRam:
    stack = []

    def __init__(self,stack_size,preprocessor):
        self.stack_size=stack_size
        self.preprocessor = preprocessor

    def reset(self,observation):
        self.stack = []
        obs = self.preprocessor.preprocess(observation)
        for i in range(self.stack_size):
            self.stack.append(obs)
        return torch.cat(self.stack,0).float()

    def step(self,observation):
        self.stack.pop(0)
        self.stack.append(self.preprocessor.preprocess(observation))
        return torch.cat(self.stack,0)
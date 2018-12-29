import torch
from DQN_Net import DQN_Net
import gym
from Utils.Stack import Stack
from Utils.Stack import StackRam
from Utils.Preprocessor import PreprocessorGray
from Utils.Preprocessor import PreprocessorRam

env = gym.make("SpaceInvaders-v0")
ramenv = gym.make("SpaceInvaders-ram-v0")

ppg = PreprocessorGray(84,84)
ppr = PreprocessorRam()

stackg = Stack(4,ppg)
stackr = StackRam(4,ppr)
pixduel_net = DQN_Net(channels=1,frame_stack=4,observation_space=(84,84),action_space=4,lr=0.00025,loss="huber",reduction="elementwise_mean",dueling=True,optim="sgd")
pix_net = DQN_Net(channels=1,frame_stack=4,observation_space=(84,84),action_space=4,lr=0.00025,loss="huber",reduction="elementwise_mean",dueling=False,optim="sgd")
ramduel_net = DQN_Net(channels=0,frame_stack=4,observation_space=128,action_space=4,lr=0.00025,loss="huber",reduction="elementwise_mean",dueling=True,optim="sgd")
ram_net = DQN_Net(channels=0,frame_stack=4,observation_space=128,action_space=4,lr=0.00025,loss="huber",reduction="elementwise_mean",dueling=False,optim="sgd")

stateg = env.reset()
stater = ramenv.reset()
inpg = stackg.reset(stateg)
inpr = stackr.reset(stater)

print(pixduel_net(inpg))
print(pix_net(inpg))
print(ramduel_net(inpr))
print(ram_net(inpr))
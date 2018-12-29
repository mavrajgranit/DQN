import torch
import torch.nn as nn
import torch.optim as opt
import torch.nn.functional as F
from Utils.Flatten import Flatten

class DQN_Net(nn.Module):

    #Change this to args and simple get args.ARGUMENT from the parser
    def __init__(self,channels=1,frame_stack=4,observation_space=(84,84),action_space=4,lr=0.00025,loss="huber",reduction="elementwise_mean",dueling=False,optim="sgd"):
        super(DQN_Net,self).__init__()
        self.channels = channels
        self.frame_stack = frame_stack
        self.observation_space = observation_space
        self.action_space = action_space
        self.lr = lr
        self.loss = loss
        self.reduction = reduction

        if loss == "huber":
            self.criterion = nn.SmoothL1Loss(reduction=reduction)
        else:
            self.criterion = nn.MSELoss(reduction=reduction)

        if reduction == "none":
            self.learn = self.learn_weighted
        else:
            self.learn = self.learn_normal

        self.dueling = dueling

        if channels>0:
            self.init_pixel_network()
            if dueling:
                self.forward = self.forward_dueling_pixel
            else:
                self.forward = self.forward_normal_pixel
        else:
            self.init_ram_network()
            if dueling:
                self.forward = self.forward_dueling_ram
            else:
                self.forward = self.forward_normal_ram

        self.optim = optim
        if optim=="rmsprop":
            self.optimizer = opt.RMSprop(self.parameters(), lr=lr)
        else:
            self.optimizer = opt.SGD(self.parameters(), lr=lr)
        self.float()

    def init_pixel_network(self):
        self.conv1 = nn.Conv2d(self.channels * self.frame_stack, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3)
        self.flatten = Flatten()
        w = ((((self.observation_space[0]-8)/4)+1-4)/2)+1-3+1
        h = ((((self.observation_space[1]-8)/4)+1-4)/2)+1-3+1
        self.fc = nn.Linear(int(w*h*64), 256)

        if self.dueling:
            self.advantage = nn.Linear(256, self.action_space)
            self.value = nn.Linear(256, 1)
        else:
            self.out = nn.Linear(256, self.action_space)

    def init_ram_network(self):
        self.fc1 = nn.Linear(self.observation_space* self.frame_stack,256)
        self.fc2 = nn.Linear(256, 128)
        if self.dueling:
            self.advantage = nn.Linear(128, self.action_space)
            self.value = nn.Linear(128, 1)
        else:
            self.out = nn.Linear(128, self.action_space)

    def forward_normal_pixel(self,input):
        x = F.relu(self.conv1(input))
        x = F.relu(self.conv2(x))
        x = self.flatten(F.relu(self.conv3(x)))
        x = F.relu(self.fc(x))
        x = self.out(x)
        return x

    def forward_dueling_pixel(self,input):
        x = F.relu(self.conv1(input))
        x = F.relu(self.conv2(x))
        x = self.flatten(F.relu(self.conv3(x)))
        x = F.relu(self.fc(x))
        adv = self.advantage(x)
        val = self.value(x)

        q = val + adv - adv.mean()
        return q

    def forward_normal_ram(self,input):
        x = F.relu(self.fc1(input))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x

    def forward_dueling_ram(self,input):
        x = F.relu(self.fc1(input))
        x = F.relu(self.fc2(x))
        adv = self.advantage(x)
        val = self.value(x)

        q = val + adv - adv.mean()
        return q

    #Normal backward-pass
    def learn_normal(self,out,target):
        self.optimizer.zero_grad()
        loss = self.criterion(out,target)
        loss.backward()
        self.optimizer.step()
        return loss

    #Prioritized Memory backward-pass
    def learn_weighted(self,out,target,weights):
        self.optimizer.zero_grad()
        loss = self.criterion(out, target)
        n = loss.shape[0]
        for i in range(len(weights)):
            loss[i] = weights[i] * loss[i].clone()
        loss = torch.sum(loss) / n
        loss.backward()
        self.optimizer.step()
        return loss
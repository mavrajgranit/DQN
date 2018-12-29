import torchvision.transforms as Transforms
import torch

class PreprocessorRam:

    def preprocess(self,observation):
        return torch.tensor(observation)

class PreprocessorGray:
    def __init__(self,h,w):
        self.h = h
        self.w = w
        self.pp = Transforms.Compose([Transforms.ToPILImage(), Transforms.Grayscale(1),Transforms.Resize((w, h)), Transforms.ToTensor()])

    def preprocess(self,observation):
        return self.pp(observation).unsqueeze(0)

class PreprocessorRGB:
    def __init__(self,h,w):
        self.h = h
        self.w = w
        self.pp = Transforms.Compose([Transforms.ToPILImage(), Transforms.Resize((w, h)), Transforms.ToTensor()])

    def preprocess(self,observation):
        return self.pp(observation).unsqueeze(0)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class CRNN(nn.Module):
    def __init__(self, imgHeight, numChannels, ):
        super(CRNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(numChannels, ), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(), nn.ReLU(), nn.MaxPool2d(2, 2),
        )

if __name__ == '__main__':
    a = 1
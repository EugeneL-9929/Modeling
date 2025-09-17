import torch
import torch.nn as nn

class LongShortTermMemory(nn.Module):
    def __init__(self, inputSize, hiddenSize, layerNum):
        super.__init__(LongShortTermMemory, self)
        self.lstmUnit = nn.LSTM(input_size=inputSize, hidden_size=hiddenSize, num_layers=layerNum)

if __name__ == '__main__':
    print(dir(torch._C))
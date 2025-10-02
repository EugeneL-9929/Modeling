import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset

class StockLSTMModel(nn.Module):
    def __init__(self, inputSize, hiddenSize, layers, outputSize, steps):
        super(StockLSTMModel, self).__init__()
        self.inputSize = inputSize
        self.steps = steps
        self.layers = layers
        self.outputSize = outputSize
        self.lstm = nn.LSTM(
            input_size=inputSize,
            hidden_size=hiddenSize,
            num_layers=layers,
            batch_first=True,
            )
        self.linear = nn.Linear(hiddenSize, outputSize)
        self.activation = nn.ELU()
    
    def forward(self, input):
        h0 = torch.zeros(self.layers, input.shape[0], self.lstm.hidden_size).requires_grad_()
        c0 = torch.zeros(self.layers, input.shape[0], self.lstm.hidden_size).requires_grad_()
        hiddenBySeries, [hiddenByLayers, cellByLayers] = self.lstm(input, (h0.detach(), c0.detach()))
        # lstmOutput = hiddenBySeries[:, -1, :]
        lstmOutput = hiddenBySeries.reshape(-1, self.lstm.hidden_size)
        activation = self.activation(lstmOutput)
        output = self.linear(activation)
        output = output.reshape(-1, self.steps, self.outputSize)
        return output[:, -1, :]

class StockLSTMOptimization():
    def __init__(self, model, trainDataset, validDataset, criterion, device):
        self.model = model
        self.trainDataset = trainDataset
        self.validDataset = validDataset
        self.criterion = criterion
        self.device = device
        self.model.to(device)
        self.optimizer = None
        self.scheduler = None
        # training history
        self.trainLosses = []
        self.validLosses = []
        self.learningRates = []
    
    def optimizerAdam(self, learningRate=0.001, **kwargs):
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learningRate,
            betas=kwargs.get('betas', (0.9, 0.999)),
            eps=kwargs.get('epsilon', 1e-8),
            weight_decay=kwargs.get('weight_devay', 0),
        )
    
    def schedulerStepLR(self, **kwargs):
        self.scheduler = StepLR(
            self.optimizer,
            step_size=kwargs.get('step_size', 30),
            gamma=kwargs.get('gamma', 0.1),
        )

    def trainEpoch(self):
        self.model.train()
        runningLoss = 0
        for batchID, (data, target) in enumerate(self.trainDataset):
            data, target = data.to(self.device), target.to(self.device)
            output = self.model(data)
            loss = self.criterion(output, target)
            runningLoss += loss.item()
            
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
        epochLoss = runningLoss/len(self.trainDataset)
        self.trainLosses.append(epochLoss)
        return epochLoss
    
    def validEpoch(self):
        self.model.eval()
        valLoss = 0
        with torch.no_grad():
            for data, target in self.validDataset:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                valLoss += self.criterion(output, target).item()
        self.validLosses.append(valLoss/len(self.validDataset))
        return valLoss
        
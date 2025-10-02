import torch.nn as nn
import torch.optim as optim

class Optimizer():
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
    
    def optimizorSGD(self, learningRate=0.01, **kwargs):
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=learningRate,
            momentum=kwargs.get('beta', 0.9),
            weight_decay=kwargs.get('weight_decay', 0.0),
        )
    
    
    def optimizorAdam(self, learningRate=0.001, **kwargs):
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learningRate,
            betas=kwargs.get('betas', (0.9, 0.999)),
            eps=kwargs.get('epsilon', 1e-8),
            weight_decay=kwargs.get('weight_devay', 0),
        )
    
    def optimizorAdagrad(self, learningRate=0.001, **kwargs):
        self.optimizer = optim.Adagrad(
            self.model.parameters(),
            lr=learningRate,
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
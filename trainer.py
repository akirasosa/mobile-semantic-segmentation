import pandas as pd
import torch


class Trainer:
    def __init__(self, data_loaders, criterion, device, on_after_epoch=None):
        self.data_loaders = data_loaders
        self.criterion = criterion
        self.device = device
        self.history = []
        self.on_after_epoch = on_after_epoch

    def train(self, model, optimizer, num_epochs):
        for epoch in range(num_epochs):
            train_epoch_loss = self._train_on_epoch(model, optimizer)
            val_epoch_loss = self._val_on_epoch(model, optimizer)

            hist = {
                'epoch': epoch,
                'train_loss': train_epoch_loss,
                'val_loss': val_epoch_loss,
            }
            self.history.append(hist)

            if self.on_after_epoch is not None:
                self.on_after_epoch(model, pd.DataFrame(self.history))

        return pd.DataFrame(self.history)

    def _train_on_epoch(self, model, optimizer):
        model.train()
        data_loader = self.data_loaders[0]
        running_loss = 0.0

        for inputs, labels in data_loader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(data_loader.dataset)

        return epoch_loss

    def _val_on_epoch(self, model, optimizer):
        model.eval()
        data_loader = self.data_loaders[1]
        running_loss = 0.0

        for inputs, labels in data_loader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(False):
                outputs = model(inputs)
                loss = self.criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(data_loader.dataset)

        return epoch_loss

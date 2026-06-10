import torch
from torch.nn import CrossEntropyLoss
from torch import optim
from services.iris_flower_data_loader import IrisDataLoader
from services.iris_flower_classification import IrisFlowerClassifier



class IrisFlowerTrainer:


    def __init__(self):
        
        self.iris_data_loader = IrisDataLoader()
        self._model = IrisFlowerClassifier()
        self._data = self.iris_data_loader.transform_data()


    def train(self, epochs:int, lr:float):

        data = self._data
        criterion = CrossEntropyLoss()
        optimizer = optim.Adam(self._model.parameters(), lr=lr)
        for epoch in range(1, epochs+1):
            self._model.train()
            predictions = self._model(data.get("X_train"))
            loss = criterion(predictions, data.get("y_train"))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if epoch % 10 == 0:

                self._model.eval()
                with torch.no_grad():
                    test_logit = self._model(data.get("X_train"))
                    pred = test_logit.argmax(dim=1)
                    acc = (pred == data.get("y_train")).float().mean() * 100
                    print(f"Epoch: {epoch} Loss: {loss.item():.4f} accuracy: {acc:.4f}")
        return self._model
    

    def evaluate(self):

        data = self._data
        self._model.eval()
        with torch.no_grad():
            logit = self._model(data.get("X_test"))
            pred = logit.argmax(dim=1)
            acc = (pred == data.get("y_test")).float().mean() * 100
            print("Accuracy: ", acc.item())
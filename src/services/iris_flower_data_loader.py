import torch
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split



class IrisDataLoader:


    def __init__(self):
        
        self._data = load_iris()


    @property
    def data(self):

        X, y = self._data.data, self._data.target
        return {
            "X": X,
            "y": y
        }
    

    def transform_data(self):

        data = self.data
        X_train, X_test, y_train, y_test = train_test_split(data.get("X"),
                                                            data.get("y"),
                                                            test_size=0.2,
                                                            random_state=0)

        X_train = torch.tensor(X_train, dtype=torch.float32)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.long)
        y_test = torch.tensor(y_test, dtype=torch.long)

        return {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test
        }
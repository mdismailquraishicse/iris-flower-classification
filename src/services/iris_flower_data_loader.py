from sklearn.datasets import load_iris



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
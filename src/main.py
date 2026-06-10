from services.iris_flower_data_loader import IrisDataLoader
from services.iris_flower_classification import IrisFlowerClassifier


def main():
    print("Hello from iris-flower-classification!")
    iris_data_loader = IrisDataLoader()
    classifier = IrisFlowerClassifier()
    data = iris_data_loader.transform_data()
    



if __name__ == "__main__":
    main()

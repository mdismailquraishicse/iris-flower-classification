from services.iris_flower_data_loader import IrisDataLoader


def main():
    print("Hello from iris-flower-classification!")
    iris_data_loader = IrisDataLoader()
    data = iris_data_loader.data
    print(data)
    print(data.keys())



if __name__ == "__main__":
    main()

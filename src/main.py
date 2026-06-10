from services.train import IrisFlowerTrainer


def main():
    print("Hello from iris-flower-classification!")
    trainer = IrisFlowerTrainer()
    trainer.train(epochs=200, lr=0.01)
    trainer.evaluate()
    



if __name__ == "__main__":
    main()

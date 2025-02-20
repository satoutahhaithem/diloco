from examples.cnn import CNNModel  # This imports code that runs 
from diloco_sim import DilocoSimulator
import torch.nn.functional as F
from examples.cnn import train_dataset, test_dataset  # Ensure 'data' module exists


def main():
    simulator = DilocoSimulator(
        model_cls=CNNModel,
        model_kwargs={"num_classes": 100},
        train_dataset=train_dataset,
        loss_fn=F.cross_entropy,
    )
    simulator.train()

if __name__ == "__main__":
    main()


# import gym

# from diloco_sim import DilocoSimulator
# from models import ModelArchitecture
# import torch.nn.functional as F
# from data import train_dataset, test_dataset

# simulator = DilocoSimulator(
#     model_cls=CNNModel,
#     model_kwargs={"num_classes": 100},
#     train_dataset=train_dataset,
#     loss_fn=F.cross_entropy,
# )

# simulator.train()


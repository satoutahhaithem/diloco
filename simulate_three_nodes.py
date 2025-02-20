from examples.cnn import CNNModel  # Correct import path
from diloco_sim.diloco import DilocoSimulator
import torch.nn.functional as F
from examples.cnn import train_dataset

# Initialize simulator with 3 nodes
simulator = DilocoSimulator(
    model_cls=CNNModel,
    model_kwargs={"num_classes": 100},
    train_dataset=train_dataset,
    loss_fn=F.cross_entropy,
    num_nodes=3
)

# Run simulation
simulator.simulate_distributed_training(num_epochs=5)

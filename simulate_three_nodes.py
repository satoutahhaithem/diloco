from examples.cnn import CNNModel, train_dataset
from diloco_sim.diloco import DilocoSimulator
import torch.nn.functional as F
import torch.nn as nn

# Modified Server Model with correct dimensions
class ServerModel(CNNModel):
    def __init__(self, num_classes):
        super().__init__(num_classes)
        # Add server-specific layers while keeping base structure
        self.extra_fc = nn.Linear(128, 256)
        self.final_fc = nn.Linear(256, num_classes)
        
    def forward(self, x):
        # Base CNN features
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 5 * 5)
        
        # Shared layers
        x = F.relu(self.fc1(x))
        
        # Server-only layers
        x = F.relu(self.extra_fc(x))
        x = self.final_fc(x)
        return x

# Initialize simulator with base configuration
simulator = DilocoSimulator(
    model_cls=CNNModel,
    model_kwargs={"num_classes": 10},
    train_dataset=train_dataset,
    loss_fn=F.cross_entropy,
    num_nodes=3
)

# Replace nodes with specialized models
simulator.nodes = [
    ServerModel(num_classes=10),  # Server
    CNNModel(num_classes=10),      # Phone
    CNNModel(num_classes=10)       # PC
]

# Run simulation
simulator.simulate_distributed_training(num_epochs=5)
simulator.simulate_distributed_training(num_epochs=5)
simulator.print_final_comparison()
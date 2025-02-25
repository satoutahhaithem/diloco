import torch
import torch.distributed as dist
from tqdm import tqdm
import torch.optim as optim
import torch.nn.utils as nn_utils
from dataclasses import dataclass
import wandb

@dataclass
class DilocoSimulatorConfig:                                            
    batch_size: int = 32                                                        
    num_epochs: int = 10
    learning_rate: float = 0.001
    num_nodes: int = 3                                                                                                                            
    device: str = 'cpu'                                 
    log_interval: int = 10                  

class DilocoSimulator:
    def __init__(self, model_cls, model_kwargs, train_dataset, loss_fn, num_nodes=3):
        self.config = DilocoSimulatorConfig()
        self.nodes = [model_cls(**model_kwargs) for _ in range(num_nodes)]
        self.train_dataset = train_dataset
        self.loss_fn = loss_fn
        self.num_nodes = num_nodes
        self.performance_history = []

    def simulate_distributed_training(self, num_epochs=10):
        self.config.num_epochs = num_epochs
        for epoch in range(self.config.num_epochs):
            epoch_metrics = {}
            
            # Train all nodes
            for node_idx, model in enumerate(self.nodes):
                device_type = self._get_device_type(node_idx)
                loss, acc = self._train_node(model, device_type)
                epoch_metrics[device_type] = {'loss': loss, 'accuracy': acc}
            self.performance_history.append(epoch_metrics)
            self._print_performance_diff(epoch_metrics)

            # Federated averaging
            self._aggregate_models()
    def print_final_comparison(self):
        final_metrics = self.performance_history[-1]
        print("\nFinal Performance Comparison:")
        for device, metrics in final_metrics.items():
            print(f"{device.capitalize()}: Loss={metrics['loss']:.4f}, Accuracy={metrics['accuracy']:.2%}")




    def _print_performance_diff(self, metrics):
        server = metrics['server']
        phone = metrics['phone']
        pc = metrics['pc']

        print("\nPerformance Differences:")
        print(f"Server vs Phone: Loss Δ={server['loss'] - phone['loss']:.4f} | Acc Δ={server['accuracy'] - phone['accuracy']:.2%}")
        print(f"Server vs PC:    Loss Δ={server['loss'] - pc['loss']:.4f} | Acc Δ={server['accuracy'] - pc['accuracy']:.2%}")
        print(f"Phone vs PC:     Loss Δ={phone['loss'] - pc['loss']:.4f} | Acc Δ={phone['accuracy'] - pc['accuracy']:.2%}")
        print("------------------------------------------------")

    def _get_device_type(self, node_idx):
        return ['server', 'phone', 'pc'][node_idx]

    def _train_node(self, model, device_type):
        model.train()
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        
        batch_size = {
            'server': 128,
            'phone': 32,
            'pc': 64
        }[device_type]

        inputs = torch.randn(batch_size, 1, 28, 28)
        labels = torch.randint(0, 10, (batch_size,))
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = self.loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        # Calculate accuracy
        _, predicted = torch.max(outputs, 1)
        correct = (predicted == labels).sum().item()
        accuracy = correct / batch_size

        print(f"{device_type} loss: {loss.item():.4f}, accuracy: {accuracy:.2%}")
        
        # Return both loss and accuracy
        return loss.item(), accuracy  # This line was missing
    def _aggregate_models(self):
        """Handle models with different architectures during aggregation"""
        with torch.no_grad():
            # Get common parameter keys across all models
            common_keys = set(self.nodes[0].state_dict().keys())
            for node in self.nodes[1:]:
                common_keys.intersection_update(node.state_dict().keys())
            
            # Average only common parameters
            averaged_params = {}
            for key in common_keys:
                params = torch.stack([node.state_dict()[key] for node in self.nodes])
                averaged_params[key] = params.mean(dim=0)

            # Update only common parameters in all models
            for node in self.nodes:
                state_dict = node.state_dict()
                for key in common_keys:
                    state_dict[key] = averaged_params[key]
                node.load_state_dict(state_dict)
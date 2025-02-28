import torch
import random
import time
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from .config import DilocoSimulatorConfig

class FederatedCarSystem:
    def __init__(self, num_classes=10):
        # Utilise la configuration centralisée
        self.config = DilocoSimulatorConfig()
        
        # Création des modèles pour chaque nœud :
        # Phone et PC utilisent un modèle léger, le Server un modèle plus puissant.
        self.phone = CarModel(num_classes)
        self.pc = CarModel(num_classes)
        self.server = ServerModel(num_classes)
        
        # Initialisation des statistiques de connexion et de latence
        self.connection_stats = {
            'pc_phone': {'success': 0, 'total': 0},
            'pc_server': {'success': 0, 'total': 0, 'latencies': []}
        }
        
        # Chargement du dataset MNIST pour l'entraînement
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        
        # Initialisation des poids des modèles
        self._init_weights(self.phone)
        self._init_weights(self.pc)
        self._init_weights(self.server)
        
        # Initialisation de la simulation de déplacement du véhicule
        self.vehicle_position = self.config.initial_vehicle_position
        self.velocity = self.config.velocity
        self.road_length = self.config.road_length

    def _init_weights(self, model):
        """Initialise les poids des couches du modèle avec des méthodes standard."""
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def _aggregate_models(self, source, target):
        """Fusionne les paramètres de deux modèles en prenant la moyenne de leurs poids."""
        source_dict = source.state_dict()
        target_dict = target.state_dict()
        merged_dict = {}
        for key in target_dict:
            if key in source_dict and source_dict[key].shape == target_dict[key].shape:
                merged_dict[key] = (source_dict[key] + target_dict[key]) / 2
        target.load_state_dict(merged_dict, strict=False)

    def _log_connection(self, connection_type, success, latency=None):
        """Enregistre le résultat d'une tentative de connexion (succès/échec) et la latence si disponible."""
        self.connection_stats[connection_type]['total'] += 1
        if success:
            self.connection_stats[connection_type]['success'] += 1
            if latency is not None:
                self.connection_stats[connection_type]['latencies'].append(latency)

    def _train_device(self, model, loader, optimizer):
        """Entraîne le modèle sur l'appareil en traitant les batches du DataLoader."""
        model.train()
        total_loss = 0
        correct = 0
        for inputs, labels in loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = nn.CrossEntropyLoss()(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
        acc = correct / len(loader.dataset)
        return total_loss / len(loader), acc

    def _simulate_pc_server_connection(self):
        """
        Simule la connexion entre le PC et le serveur.
        La probabilité de réussite dépend de la position du véhicule :
          - En zone blanche (position >= 50), la déconnexion est plus probable.
          - Sinon, la connexion est plus fiable.
        La latence est simulée si activée.
        """
        if self.vehicle_position >= 50.0:
            dynamic_dropout = 0.8  # 80% de chance d'échec en zone blanche
        else:
            dynamic_dropout = 0.1  # 10% de chance d'échec en zone de bonne couverture
        
        start_time = time.time()
        if self.config.simulate_latency:
            simulated_delay = random.uniform(*self.config.latency_range)
            time.sleep(simulated_delay)
        connection_success = random.random() > dynamic_dropout
        elapsed_time = time.time() - start_time
        return connection_success, elapsed_time

    def _update_vehicle_position(self):
        """
        Met à jour la position du véhicule en fonction de sa vitesse.
        La position est cyclique : lorsque le véhicule atteint la fin de la route,
        il recommence au début.
        """
        self.vehicle_position = (self.vehicle_position + self.velocity) % self.road_length

    def run_training(self):
        """Boucle d'entraînement principale orchestrant la synchronisation et l'entraînement des modèles."""
        train_loader = DataLoader(self.train_data, batch_size=self.config.batch_size, shuffle=True)
        phone_optim = optim.SGD(self.phone.parameters(), lr=0.01, momentum=0.9)
        pc_optim = optim.SGD(self.pc.parameters(), lr=0.01, momentum=0.9)
        server_optim = optim.Adam(self.server.parameters(), lr=0.001)
        
        for epoch in range(self.config.num_epochs):
            # Mise à jour de la position du véhicule
            self._update_vehicle_position()
            print(f"Position du véhicule: {self.vehicle_position:.2f}")
            
            # Synchronisation PC-Phone : toujours réussie
            self._aggregate_models(self.phone, self.pc)
            self._log_connection('pc_phone', True)
            
            # Entraînement local sur chaque appareil
            phone_loss, phone_acc = self._train_device(self.phone, train_loader, phone_optim)
            pc_loss, pc_acc = self._train_device(self.pc, train_loader, pc_optim)
            server_loss, server_acc = self._train_device(self.server, train_loader, server_optim)
            
            # Simulation de la connexion PC-Server en fonction de la position
            success, latency = self._simulate_pc_server_connection()
            if success:
                self._aggregate_models(self.pc, self.server)
                pc_server_status = "✓"
                self._log_connection('pc_server', True, latency)
            else:
                pc_server_status = "✗"
                self._log_connection('pc_server', False)
            
            # Affichage des résultats pour l'époque en cours
            print(f"\nEpoch {epoch+1}/{self.config.num_epochs}")
            print("┌──────────────────────┬──────────────────────┐")
            print(f"│ PC-Phone Connection  │       ✓          │")
            print(f"│ PC-Server Connection │       {pc_server_status}          │")
            print("└──────────────────────┴──────────────────────┘")
            print(f"Phone Model │ Loss: {phone_loss:.4f} │ Accuracy: {phone_acc:.2%}")
            print(f"PC Model    │ Loss: {pc_loss:.4f} │ Accuracy: {pc_acc:.2%}")
            print(f"Server Model│ Loss: {server_loss:.4f} │ Accuracy: {server_acc:.2%}")
            print("─" * 50)
        
        # Rapport final de qualité de service
        print("\n=== Rapport Final de Connectivité ===")
        print(f"Connexions PC-Phone réussies: {self.connection_stats['pc_phone']['success']}/{self.connection_stats['pc_phone']['total']}")
        print(f"Connexions PC-Serveur réussies: {self.connection_stats['pc_server']['success']}/{self.connection_stats['pc_server']['total']}")
        success_rate = (self.connection_stats['pc_server']['success'] / self.connection_stats['pc_server']['total']
                        if self.connection_stats['pc_server']['total'] else 0)
        print(f"Taux de réussite PC-Serveur: {success_rate:.2%}")
        if self.connection_stats['pc_server']['latencies']:
            avg_latency = sum(self.connection_stats['pc_server']['latencies']) / len(self.connection_stats['pc_server']['latencies'])
            print(f"Latence moyenne PC-Serveur: {avg_latency:.3f} secondes")

# Modèle léger pour Phone et PC
class CarModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        return self.classifier(x.view(x.size(0), -1))

# Modèle plus puissant pour le Server
class ServerModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 7 * 7, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        return self.classifier(x.view(x.size(0), -1))

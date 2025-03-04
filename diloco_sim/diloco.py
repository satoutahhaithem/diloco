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
        # Le Phone et le PC utilisent un modèle léger, le Server un modèle plus puissant.
        self.phone = CarModel(num_classes)
        self.pc = CarModel(num_classes)
        self.server = ServerModel(num_classes)
        
        # Initialisation des statistiques de connexion et latence pour la communication inter-nœuds
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

    def _distribute_tasks(self, inputs, distribution):
        """
        Divise les indices d'un batch selon la distribution souhaitée.
        :param inputs: Batch d'inputs.
        :param distribution: Dictionnaire, par exemple {'server': 0.5, 'pc': 0.3, 'phone': 0.2}.
        :return: Un dictionnaire de sous-indices pour chaque nœud.
        """
        batch_size = inputs.size(0)
        indices = torch.randperm(batch_size)
        splits = {}
        start = 0
        keys = list(distribution.keys())
        for i, node in enumerate(keys):
            if i == len(keys) - 1:
                count = batch_size - start
            else:
                count = int(distribution[node] * batch_size)
            splits[node] = indices[start:start+count]
            start += count
        return splits

    def _train_device(self, model, inputs, labels, optimizer):
        """
        Entraîne le modèle sur un sous-batch et retourne la loss, le nombre de bonnes prédictions,
        le nombre d'échantillons traités et la latence de traitement.
        """
        start_time = time.time()
        model.train()
        outputs = model(inputs)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        end_time = time.time()
        elapsed = end_time - start_time
        _, preds = torch.max(outputs, 1)
        correct = (preds == labels).sum().item()
        return loss.item() * inputs.size(0), correct, inputs.size(0), elapsed

    def _simulate_pc_server_connection(self):
        """
        Simule la connexion entre le PC et le serveur en fonction de la position du véhicule.
        - Position < 40: bonne couverture (faible probabilité d'échec)
        - 40 <= Position < 60: zone grise (probabilité intermédiaire)
        - Position >= 60: faible couverture (forte probabilité d'échec)
        La latence est simulée si activée.
        """
        if self.vehicle_position < 40.0:
            dynamic_dropout = 0.1  # 10% de chance d'échec
        elif self.vehicle_position < 60.0:
            dynamic_dropout = 0.5  # 50% de chance d'échec (zone grise)
        else:
            dynamic_dropout = 0.8  # 80% de chance d'échec

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
        La position est cyclique : quand le véhicule atteint la fin, il recommence.
        """
        self.vehicle_position = (self.vehicle_position + self.velocity) % self.road_length

    def run_training(self):
        """
        Boucle d'entraînement principale qui orchestre la répartition des tâches, la synchronisation et l'entraînement.
        Calcule et affiche les métriques globales de performance (loss, accuracy, samples et latence moyenne) pour l'ensemble du cluster.
        """
        train_loader = DataLoader(self.train_data, batch_size=self.config.batch_size, shuffle=True)
        phone_optim = optim.SGD(self.phone.parameters(), lr=0.01, momentum=0.9)
        pc_optim = optim.SGD(self.pc.parameters(), lr=0.01, momentum=0.9)
        server_optim = optim.Adam(self.server.parameters(), lr=0.001)
        
        for epoch in range(self.config.num_epochs):
            self._update_vehicle_position()
            print(f"\nPosition du véhicule: {self.vehicle_position:.2f}")
            
            # Détermination de la répartition en fonction de la connectivité PC-Server
            connectivity, comm_latency = self._simulate_pc_server_connection()
            if connectivity:
                distribution = {'server': 0.5, 'pc': 0.3, 'phone': 0.2}
                pc_server_status = "✓"
                self._log_connection('pc_server', True, comm_latency)
            else:
                distribution = {'pc': 0.6, 'phone': 0.4}  # Le serveur n'est pas utilisé
                pc_server_status = "✗"
                self._log_connection('pc_server', False)
            
            # Synchronisation PC-Phone (toujours réussie)
            self._aggregate_models(self.phone, self.pc)
            self._log_connection('pc_phone', True)
            
            # Initialisation des accumulateurs de métriques pour l'époque
            phone_total_loss = 0.0; phone_correct = 0; phone_samples = 0; phone_latency = 0.0
            pc_total_loss = 0.0; pc_correct = 0; pc_samples = 0; pc_latency = 0.0
            server_total_loss = 0.0; server_correct = 0; server_samples = 0; server_latency = 0.0
            
            for inputs, labels in train_loader:
                splits = self._distribute_tasks(inputs, distribution)
                
                if 'phone' in splits and splits['phone'].numel() > 0:
                    phone_inputs = inputs[splits['phone']]
                    phone_labels = labels[splits['phone']]
                    loss_val, correct_val, n_samples, lat = self._train_device(self.phone, phone_inputs, phone_labels, phone_optim)
                    phone_total_loss += loss_val
                    phone_correct += correct_val
                    phone_samples += n_samples
                    phone_latency += lat
                
                if 'pc' in splits and splits['pc'].numel() > 0:
                    pc_inputs = inputs[splits['pc']]
                    pc_labels = labels[splits['pc']]
                    loss_val, correct_val, n_samples, lat = self._train_device(self.pc, pc_inputs, pc_labels, pc_optim)
                    pc_total_loss += loss_val
                    pc_correct += correct_val
                    pc_samples += n_samples
                    pc_latency += lat
                
                if 'server' in splits and splits.get('server') is not None and splits['server'].numel() > 0:
                    server_inputs = inputs[splits['server']]
                    server_labels = labels[splits['server']]
                    loss_val, correct_val, n_samples, lat = self._train_device(self.server, server_inputs, server_labels, server_optim)
                    server_total_loss += loss_val
                    server_correct += correct_val
                    server_samples += n_samples
                    server_latency += lat
            
            # Calcul des métriques par nœud
            phone_loss_avg = phone_total_loss / phone_samples if phone_samples > 0 else 0
            phone_acc = phone_correct / phone_samples if phone_samples > 0 else 0
            phone_latency_avg = phone_latency / (phone_samples / self.config.batch_size) if phone_samples > 0 else 0

            pc_loss_avg = pc_total_loss / pc_samples if pc_samples > 0 else 0
            pc_acc = pc_correct / pc_samples if pc_samples > 0 else 0
            pc_latency_avg = pc_latency / (pc_samples / self.config.batch_size) if pc_samples > 0 else 0

            server_loss_avg = server_total_loss / server_samples if server_samples > 0 else 0
            server_acc = server_correct / server_samples if server_samples > 0 else 0
            server_latency_avg = server_latency / (server_samples / self.config.batch_size) if server_samples > 0 else 0
            
            # Calcul des métriques globales pondérées
            total_samples = phone_samples + pc_samples + server_samples
            global_loss = (phone_total_loss + pc_total_loss + server_total_loss) / total_samples if total_samples > 0 else 0
            global_correct = phone_correct + pc_correct + server_correct
            global_acc = global_correct / total_samples if total_samples > 0 else 0
            
            # Affichage des métriques de l'époque
            print(f"\nEpoch {epoch+1}/{self.config.num_epochs}")
            print("┌─────────────────────────────┬─────────────────────────────┐")
            print(f"│ PC-Phone Connection         │       ✓                     │")
            print(f"│ PC-Server Connection        │       {pc_server_status}                     │")
            print("└─────────────────────────────┴─────────────────────────────┘")
            print(f"Phone Model    : Loss: {phone_loss_avg:.4f}, Accuracy: {phone_acc:.2%}, Samples: {phone_samples}, Latency: {phone_latency_avg:.3f}s")
            print(f"PC Model       : Loss: {pc_loss_avg:.4f}, Accuracy: {pc_acc:.2%}, Samples: {pc_samples}, Latency: {pc_latency_avg:.3f}s")
            print(f"Server Model   : Loss: {server_loss_avg:.4f}, Accuracy: {server_acc:.2%}, Samples: {server_samples}, Latency: {server_latency_avg:.3f}s")
            print(f"Global Cluster : Loss: {global_loss:.4f}, Accuracy: {global_acc:.2%}, Samples: {total_samples}")
            print("─" * 50)
        
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
# exo , mon trvavail, exo gym 
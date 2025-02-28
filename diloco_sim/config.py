from dataclasses import dataclass, field
from typing import Optional, Callable, Type
import torch

@dataclass
class DilocoSimulatorConfig:
    # Paramètres liés au modèle
    model_cls: Type[torch.nn.Module] = None
    model_kwargs: dict = field(default_factory=dict)
    loss_fn: Callable[..., torch.Tensor] = torch.nn.CrossEntropyLoss()
    
    # Datasets
    train_dataset: torch.utils.data.Dataset = None
    eval_dataset: Optional[torch.utils.data.Dataset] = None

    # Optimiseur
    optimizer_kwargs: dict = field(default_factory=dict)
    optimizer_cls: Type[torch.optim.Optimizer] = torch.optim.AdamW
    
    # Paramètres d'entraînement
    batch_size: int = 128
    num_epochs: int = 10

    # Paramètres additionnels pour la simulation fédérée
    p_sparta: float = 0.0
    sparta_interval: int = 1
    cosine_anneal: bool = False
    warmup_steps: int = 0
    model_path: Optional[str] = None
    num_nodes: int = 3  # Trois nœuds : Phone, PC et Server
    diloco_interval: int = 500

    # Optimiseur externe pour la synchronisation (ex : entre PC et Server)
    outer_optimizer_cls: Type[torch.optim.Optimizer] = torch.optim.SGD
    outer_optimizer_kwargs: dict = field(default_factory=lambda: {"lr": 0.7, "nesterov": True, "momentum": 0.9})
    
    max_local_step: Optional[int] = None
    wandb_project: Optional[str] = None
    max_minibatch_size: Optional[int] = None
    port: int = 12355
    devices: Optional[list[int]] = None
    max_norm: Optional[float] = None
    async_sparta_delay: int = 0
    
    # Paramètres pour la simulation de connexion
    pc_server_dropout: float = 0.3  # 30% de chance de déconnexion PC-Server
    simulate_latency: bool = True   # Simule un délai de connexion
    latency_range: tuple = (0.05, 0.2)  # Délai simulé entre 0.05 et 0.2 secondes pour PC-Server
    
    # Paramètres pour la simulation de déplacement du véhicule
    initial_vehicle_position: float = 0.0
    velocity: float = 10.0         # Unités par époque
    road_length: float = 100.0     # Longueur totale de la route

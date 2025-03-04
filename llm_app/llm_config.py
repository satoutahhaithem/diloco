from dataclasses import dataclass

@dataclass
class LLMConfig:
    # Paramètres pour le modèle LLM local
    model: str = "distilgpt2"  # Utilise un modèle léger de GPT-2
    temperature: float = 0.7

    # Paramètres pour la simulation fédérée
    num_nodes: int = 3  # Trois nœuds : phone, pc, server 
    # Paramètres pour la simulation de connectivité PC-Server
    pc_server_dropout: float = 0.3  # 30% de chance d'échec
    simulate_latency: bool = True
    latency_range: tuple = (0.05, 0.2)  # Délai de 50ms à 200ms

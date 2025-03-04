import random
import time
from llm_config import LLMConfig
from llm_client import get_llm_response

class FederatedLLMSystem:
    def __init__(self):
        self.config = LLMConfig()
        self.nodes = ["phone", "pc", "server"]
    
    def _simulate_connectivity(self) -> bool:
        """
        Simule la connectivité entre le PC et le serveur.
        Retourne True si la connexion est établie, False sinon.
        """
        return random.random() > self.config.pc_server_dropout

    def _simulate_latency(self) -> float:
        """Simule la latence si activée, retourne le délai en secondes."""
        if self.config.simulate_latency:
            delay = random.uniform(*self.config.latency_range)
            time.sleep(delay)
            return delay
        return 0.0

    def run(self, prompt: str):
        """
        Exécute le système fédéré.
        Chaque nœud interroge le modèle LLM avec le prompt.
        Si la connexion PC-Server échoue, le serveur est exclu.
        Les réponses et la latence de chaque nœud sont affichées.
        """
        print("Starting Federated LLM System...")
        # Simuler la connectivité PC-Server
        pc_server_connected = self._simulate_connectivity()
        if pc_server_connected:
            distribution = {"phone": 0.33, "pc": 0.33, "server": 0.34}
            print("PC-Server Connection: ✓")
        else:
            distribution = {"phone": 0.5, "pc": 0.5}  # Le serveur est exclu
            print("PC-Server Connection: ✗ (server excluded)")
        
        # Pour chaque nœud participant, simuler un appel à l'API LLM
        responses = {}
        latencies = {}
        for node in distribution:
            print(f"Node '{node}' processing...")
            latency = self._simulate_latency()
            latencies[node] = latency
            response = get_llm_response(prompt)
            responses[node] = response
        
        # Affichage des résultats de chaque nœud
        print("\nFederated LLM System Results:")
        for node in responses:
            print(f"Node: {node}")
            print(f"Latency: {latencies[node]:.3f}s")
            print("Response:")
            print(responses[node])
            print("-" * 50)

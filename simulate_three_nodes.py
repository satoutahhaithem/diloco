from diloco_sim.diloco import FederatedCarSystem

if __name__ == "__main__":
    print("Démarrage du système fédéré véhicule-serveur...")
    system = FederatedCarSystem(num_classes=10)
    system.run_training()
    print("\nEntraînement terminé avec succès!")

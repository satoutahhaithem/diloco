from llm_federated import FederatedLLMSystem

if __name__ == "__main__":
    prompt = "Explain the concept of federated learning in simple terms."
    federated_system = FederatedLLMSystem()
    federated_system.run(prompt)

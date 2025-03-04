from transformers import pipeline
from llm_config import LLMConfig

# Créez une instance de configuration
config = LLMConfig()

# Instanciez le pipeline de génération de texte en utilisant le modèle défini dans la configuration
generator = pipeline("text-generation", model=config.model)

def get_llm_response(prompt: str) -> str:
    """Obtient une réponse générée par le modèle en fonction du prompt."""
    response = generator(prompt, max_length=100, do_sample=True, temperature=config.temperature)
    return response[0]["generated_text"]

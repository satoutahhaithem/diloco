import streamlit as st
from llm_client import get_llm_response  # Assurez-vous que ce fichier est accessible

st.title("Interface de Chat avec LLM")

# Zone de saisie pour l'utilisateur
user_input = st.text_input("Votre message:")

if st.button("Envoyer"):
    if user_input:
        st.write("Envoi en cours...")
        # Obtenir la réponse du LLM (via l'API ou votre pipeline HuggingFace)
        response = get_llm_response(user_input)
        st.write("Réponse:", response)

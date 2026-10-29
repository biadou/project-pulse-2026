import os
import gradio as gr
from google import genai

# Initialisation du client Gemini
api_key = os.environ.get("GEMINI_API_KEY")

SYSTEM_INSTRUCTIONS = {
    "Planner": "Tu es l'agent Planner. Tu aides à concevoir des plans d'action, des cours structurés et des feuilles de route en gardant le contexte.",
    "Coder": "Tu es l'agent Coder. Tu fournis du code technique propre, optimisé et commenté en te souvenant des échanges précédents.",
    "Designer": "Tu es l'agent Designer UI/UX. Tu conçois l'ergonomie, la structure d'interface et l'expérience utilisateur en conservant le fil.",
    "Orchestrator": "Tu es l'agent Orchestrator. Tu supervises les échanges, coordonnes les rôles et synthétises les avancées du projet."
}

def respond(message, history, agent_choice):
    if not message.strip():
        return "", history

    if not api_key:
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": "❌ Erreur : Clé GEMINI_API_KEY introuvable dans l'environnement."})
        return "", history

    try:
        client = genai.Client(api_key=api_key)
        system_instruction = SYSTEM_INSTRUCTIONS.get(agent_choice, "Tu es un assistant IA utile.")

        # Conversion sécurisée de l'historique au format attendu par le SDK Google GenAI
        chat_history = []
        for msg in history:
            role = msg.get("role")
            content = msg.get("content")
            
            # Extraction du texte si Gradio passe une liste d'objets/dictionnaires
            if isinstance(content, list):
                text_content = ""
                for part in content:
                    if isinstance(part, dict) and "text" in part:
                        text_content += part["text"]
                    elif isinstance(part, str):
                        text_content += part
                content = text_content

            # Formatage strict de 'parts' sous forme de liste de dictionnaires avec clé 'text'
            formatted_parts = [{"text": str(content)}]

            if role == "user":
                chat_history.append({"role": "user", "parts": formatted_parts})
            elif role == "assistant":
                chat_history.append({"role": "model", "parts": formatted_parts})

        # Création de la session de chat avec l'historique nettoyé
        chat = client.chats.create(
            model='gemini-3.8-flash',
            history=chat_history,
            config={"system_instruction": system_instruction}
        )

        response = chat.send_message(message)
        
        # Mise à jour de l'historique Gradio
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": response.text})
        return "", history

    except Exception as e:
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": f"❌ Erreur technique : {str(e)}"})
        return "", history

def clear_conversation():
    return [], ""

# --- Interface Gradio sur-mesure ---
with gr.Blocks(title="Mona's Project Pulse", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🚀 Mona's Project Pulse — Dashboard Conversationnel Multi-Agents")
    gr.Markdown("Sélectionnez votre agent, discutez en toute fluidité avec un historique stable.")

    with gr.Row():
        with gr.Column(scale=1):
            agent_selector = gr.Dropdown(
                choices=["Planner", "Coder", "Designer", "Orchestrator"],
                value="Planner",
                label="Sélectionner l'agent actif",
                interactive=True
            )
            gr.Markdown("---")
            clear_btn = gr.Button("🗑️ Effacer la discussion", variant="secondary")

        with gr.Column(scale=3):
            chatbot = gr.Chatbot(label="Fil de discussion", height=450)
            
            with gr.Row():
                user_input = gr.Textbox(
                    placeholder="Écrivez votre message ici...",
                    lines=2,
                    label="Votre message",
                    scale=4
                )
                submit_btn = gr.Button("Envoyer 🚀", variant="primary", scale=1)

    submit_btn.click(
        fn=respond,
        inputs=[user_input, chatbot, agent_selector],
        outputs=[user_input, chatbot]
    )
    user_input.submit(
        fn=respond,
        inputs=[user_input, chatbot, agent_selector],
        outputs=[user_input, chatbot]
    )
    clear_btn.click(fn=clear_conversation, outputs=[chatbot, user_input])

if __name__ == "__main__":
    demo.launch()

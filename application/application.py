import os
import tempfile
from typing import List, Dict, Any, Tuple

import httpx
from dotenv import load_dotenv
import gradio as gr
import asyncio

# Load environment variables
load_dotenv()

from immich_tools import IMMICH_SERVER_URL, IMMICH_API_KEY
from immich_agent import ImmichAgent

agent = ImmichAgent()

def download_thumbnail(asset_id: str) -> str | None:
    """Download an asset thumbnail from Immich and save it to a temporary file."""
    if not IMMICH_API_KEY:
        return None
    
    server_url = IMMICH_SERVER_URL.rstrip('/')
    url = f"{server_url}/api/assets/{asset_id}/thumbnail"
    
    headers = {
        "x-api-key": IMMICH_API_KEY,
        "Accept": "image/*",
    }
    
    try:
        with httpx.Client(timeout=30) as client:
            resp = client.get(url, headers=headers, params={"size": "preview"})
            if resp.status_code == 200:
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
                tmp.write(resp.content)
                tmp.flush()
                tmp.close()
                return tmp.name
    except Exception as e:
        print(f"Error downloading thumbnail for {asset_id}: {e}")
        
    return None

async def handle_stream(stream_generator, history: List[dict]):
    """
    A general-purpose generator that consumes any stream from ImmichAgent (run or resume)
    and updates the Gradio interface (text, confirmation buttons, and photo gallery).
    """
    needs_confirm = False
    found_assets = []

    try:
        accumulated_text = ""
        async for chunk in stream_generator:
            ctype = chunk.get("type")

            if ctype == "token":
                accumulated_text += chunk.get("content", "")
                history[-1]["content"] = accumulated_text
                yield history, gr.update(visible=False)

            elif ctype == "interrupt":
                data = chunk.get("data", {})
                history[-1]["content"] = data.get("response_text", "")
                needs_confirm = True
                found_assets = data.get("extracted_assets", [])
                yield history, gr.update(visible=True)

            elif ctype == "final":
                data = chunk.get("data", {})
                history[-1]["content"] = data.get("response_text", "")
                found_assets = data.get("extracted_assets", [])
                yield history, gr.update(visible=False)

    except Exception as e:
        history[-1]["content"] = f"Eroare la procesare: {e}"
        yield history, gr.update(visible=False)
        return

    # Gallery build, (executed once, if pictures exists)
    if found_assets:
        gallery_items: List[Tuple[str, str]] = []
        for i, a in enumerate(found_assets[:15], 1):
            fname = a.get("originalFileName", "Imagine")
            aid = a.get("id", "")
            created = (a.get("fileCreatedAt") or "")[:10]

            if aid:
                thumb_path = download_thumbnail(aid)
                if thumb_path:
                    caption = f"{i}. {fname}\n📅 {created}"
                    gallery_items.append((thumb_path, caption))

        if gallery_items:
            history.append({
                "role": "assistant",
                "content": gr.Gallery(
                    value=gallery_items,
                    columns=5,
                    rows=None,
                    height="auto",
                    allow_preview=True,
                    preview=True,
                    show_label=False,
                    object_fit="cover",
                )
            })
            yield history, gr.update(visible=needs_confirm)

async def chat_fn(message: str, history: List[dict]):
    """Handler for new messages of the user."""
    if not message.strip():
        yield "", history, gr.update(visible=False)
        return

    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": "..."})
    yield "", history, gr.update(visible=False)

    # Delegates whole process to handle_stram
    async for updated_history, confirm_box in handle_stream(agent.run(message), history):
        yield "", updated_history, confirm_box


async def approve_fn(history: List[dict]):
    """Handler when user clicks Approve button with streaming."""
    yield history, gr.update(visible=False)

    history.append({"role": "user", "content": "👉 *Confirm command*"})
    history.append({"role": "assistant", "content": "..."})
    yield history, gr.update(visible=False)

    async for updated_history, confirm_box in handle_stream(
        agent.resume_confirmation(approved=True), history
    ):
        yield updated_history, confirm_box

async def reject_fn(history: List[dict]):
    """Handler for cancel button."""

    history.append({"role": "user", "content": "👉 *Anulează acțiunea*"})
    history.append({"role": "assistant", "content": "..."})
    yield history, gr.update(visible=False)

    async for updated_history, confirm_box in handle_stream(agent.resume_confirmation(approved=False), history):
        yield updated_history, confirm_box

CUSTOM_CSS = """
.gradio-container {
    max-width: 100% !important;
    width: 100% !important;
    padding-left: 20px !important;
    padding-right: 20px !important;
}
.message-bubble-border, .bubble-wrap {
    max-width: 100% !important;
    width: 100% !important;
}
.gallery-container {
    width: 100% !important;
}
.confirmation-box {
    background-color: var(--block-background-fill, #fff3cd) !important;
    border: 1px solid var(--border-color-accent, #ffeeba) !important;
    padding: 12px;
    border-radius: 8px;
}
"""


def build_ui():
    """Build the Gradio interface with Human-in-the-Loop controls."""
    azure_configured = bool(os.getenv("AZURE_OPENAI_API_KEY") and os.getenv("AZURE_OPENAI_ENDPOINT"))
    model_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME") or os.getenv("AZURE_OPENAI_MODEL_NAME") or "gpt-4o"
    status_badge = f"🟢 **LLM:** Azure OpenAI (`{model_name}`) | 🛡️ **Human-in-the-Loop:** Activ" if azure_configured else "🟡 **LLM:** Fallback Rule-based | 🛡️ **Human-in-the-Loop:** Activ"

    with gr.Blocks(title="Immich AI Assistant", fill_width=True) as demo:
        gr.Markdown(
            f"""
            # 📸 Asistent Immich + LangChain (Human-in-the-Loop)
            {status_badge}
            
            Intelligent assistant for searching and organizing within albums. Actions involving creation or modification require your confirmation before being executed!
            
            **Exemple:**
            - *"Crete an album and put there the pictures with cats"*
            """
        )

        chatbot = gr.Chatbot(
            height=680,
        )

        # Human-in-the-Loop Confirmation Row (initially hidden)
        with gr.Row(visible=False, elem_classes=["confirmation-box"]) as confirm_row:
            confirm_label = gr.Markdown("### ⚠️ Confirmare acțiune: Dorești ca asistentul să execute acțiunea de mai sus?")
            with gr.Row():
                approve_btn = gr.Button("✅ Confirmă și Creează Albumul", variant="primary", scale=1)
                reject_btn = gr.Button("❌ Anulează", variant="stop", scale=1)

        with gr.Row():
            msg_input = gr.Textbox(
                placeholder="Scrie o comandă (ex: 'fa un album separat si pune acolo o poza cu pisica')...",
                show_label=False,
                scale=9,
                lines=1,
                autofocus=True,
            )
            submit_btn = gr.Button("Trimite", variant="primary", scale=1)
            
        with gr.Row():
            clear_btn = gr.ClearButton([msg_input, chatbot], value="Șterge conversația")

        # Submit handlers
        msg_input.submit(
            chat_fn,
            inputs=[msg_input, chatbot],
            outputs=[msg_input, chatbot, confirm_row]
        )
        submit_btn.click(
            chat_fn,
            inputs=[msg_input, chatbot],
            outputs=[msg_input, chatbot, confirm_row]
        )

        # Decision handlers
        approve_btn.click(
            approve_fn,
            inputs=[chatbot],
            outputs=[chatbot, confirm_row]
        )
        reject_btn.click(
            reject_fn,
            inputs=[chatbot],
            outputs=[chatbot, confirm_row]
        )

    return demo

if __name__ == "__main__":
    demo = build_ui()
    demo.launch(
        theme=gr.themes.Soft(),
        css=CUSTOM_CSS,
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )

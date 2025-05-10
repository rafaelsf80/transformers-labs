""" Simple chat with gemini-2.0-flash on Gradio
"""

import gradio as gr

from google import genai
from google.genai.types import GenerateContentConfig
import vertexai

# TODO: Change PROJECT_ID
PROJECT_ID = "YOUR_PROJECT_ID" # <--- CHANGE THIS
LOCATION = "europe-west4"
MODEL_GOOGLE = "gemini-2.0-flash"

vertexai.init(project=PROJECT_ID, location=LOCATION)

client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)

chat = client.chats.create(
    model=MODEL_GOOGLE,
    config=GenerateContentConfig(
        system_instruction="You are an helpful assistant."
    ),
)

def add_text(history, text):
    history = history + [(text, None)]
    return history, ""

def add_file(history, file):
    history = history + [((file.name,), None)]
    return history

def bot(history):
    print(history)
    text_response = chat.send_message(str(history[-1][0]))
    print(text_response.text)
    history[-1][1] = str(text_response.text)
    print(history)
    return history

with gr.Blocks() as io:
    gr.Markdown(
        """
    # gemini-2.0-flash
    ## This demo shows  chat with gemini-2.0-flash
    """
    )

    chatbot = gr.Chatbot([], elem_id="chatbot")

    with gr.Row():
        with gr.Column(scale=0.85):
            txt = gr.Textbox(
                show_label=False,
                placeholder="Enter text and press enter, or upload an image",
            )
        with gr.Column(scale=0.15, min_width=0):
            btn = gr.UploadButton("📁", file_types=["image", "video", "audio"])

    txt.submit(add_text, [chatbot, txt], [chatbot, txt]).then(
        bot, chatbot, chatbot
    )
    btn.upload(add_file, [chatbot, btn], [chatbot]).then(
        bot, chatbot, chatbot
    )

io.launch(server_name="0.0.0.0", server_port=7860, share=True)
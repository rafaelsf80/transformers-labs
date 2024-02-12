""" chat-bison@001 Gradio demo 
"""

import gradio as gr

import vertexai
from vertexai.preview.language_models import ChatModel

# TODO: Change PROJECT_ID
PROJECT_ID = "YOUR_PROJECT_ID" # <--- CHANGE THIS
LOCATION = "us-central1" 

vertexai.init(project=PROJECT_ID, location=LOCATION)

chat_model = ChatModel.from_pretrained("chat-bison@001")

chat = chat_model.start_chat()

def add_text(history, text):
    history = history + [(text, None)]
    return history, ""

def add_file(history, file):
    history = history + [((file.name,), None)]
    return history

def bot(history):
    print(history)
    text_response = chat.send_message(str(history[-1][0]))
    print(text_response)
    history[-1][1] = str(text_response)
    print(history)
    return history

with gr.Blocks() as io:
    gr.Markdown(
        """
    # chat-bison@001
    ## This demo shows chat-bison
    """
    )

    chatbot = gr.Chatbot([], elem_id="chatbot").style(height=750)

    with gr.Row():
        with gr.Column(scale=0.85):
            txt = gr.Textbox(
                show_label=False,
                placeholder="Enter text and press enter, or upload an image",
            ).style(container=False)
        with gr.Column(scale=0.15, min_width=0):
            btn = gr.UploadButton("📁", file_types=["image", "video", "audio"])

    txt.submit(add_text, [chatbot, txt], [chatbot, txt]).then(
        bot, chatbot, chatbot
    )
    btn.upload(add_file, [chatbot, btn], [chatbot]).then(
        bot, chatbot, chatbot
    )

io.launch(server_name="0.0.0.0", server_port=7860)
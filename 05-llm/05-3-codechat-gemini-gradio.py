import gradio as gr

from google import genai
from google.genai.types import Tool, ToolCodeExecution, GenerateContentConfig
import vertexai

# TODO: Change PROJECT_ID
PROJECT_ID = "YOUR_PROJECT_ID" # <--- CHANGE THIS
LOCATION = "europe-west4"
MODEL_GOOGLE = "gemini-2.0-flash"

vertexai.init(project=PROJECT_ID, location=LOCATION)

client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)

code_execution_tool = Tool(
    code_execution=ToolCodeExecution()
)

code_chat = client.chats.create(
    model=MODEL_GOOGLE,
    config=GenerateContentConfig(
        system_instruction="You are an expert software engineer, proficient in Python."
    ),
        config=GenerateContentConfig(
        tools=[code_execution_tool],
        temperature=0,
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
    text_response = code_chat.send_message(str(history[-1][0]))
    print(text_response.text)
    history[-1][1] = str(text_response.text)
    print(history)
    return history

with gr.Blocks() as io:
    gr.Markdown(
        """
    # gemini-2.0-flash
    ## This demo shows CODE EXECUTION chat with gemini-2.0-flash
    """
    )

    chatbot = gr.Chatbot([], elem_id="codechat")

    with gr.Row():
        with gr.Column(scale=0.85):
            txt = gr.Textbox(
                show_label=False,
                placeholder="Enter code and press enter, or upload a file",
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



prompt = """Generate a Docker script to create a simple linux machine 
         that has python 3.10 installed with following libraries: pandas, tensorflow, numpy"""
""" text-bison@001 Gradio demo 
"""

import gradio as gr

import vertexai
from vertexai.preview.language_models import TextGenerationModel

# TODO: Change PROJECT_ID
PROJECT_ID = "YOUR_PROJECT_ID" # <--- CHANGE THIS
LOCATION = "us-central1" 

vertexai.init(project=PROJECT_ID, location=LOCATION)

model = TextGenerationModel.from_pretrained("text-bison@001")

def predict(prompt, max_output_tokens, temperature, top_p, top_k):
    answer = model.predict(
        prompt,
        max_output_tokens=max_output_tokens, #128
        temperature=temperature,#0
        top_p=top_p, #1
        top_k=top_k) #40
    return answer

demo = gr.Interface(
    predict, 
    [ gr.Textbox(label="Enter prompt:", value="Best receipt for banana bread:"),
      gr.Slider(32, 512, value=128, step = 8, label = "max_output_tokens"),
      gr.Slider(0, 1, value=0, step = 0.1, label = "temperature"),
      gr.Slider(1, 5, value=1, step = 1, label = "top_p"),
      gr.Slider(20, 400, value=40, step = 10, label = "top_k"),
    ],
    "text"
    )

demo.launch(server_name="0.0.0.0", server_port=7860)
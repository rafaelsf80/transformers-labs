""" Simple prompting Gradio demo 
"""

import gradio as gr

from google import genai
from google.genai.types import GenerateContentConfig
import vertexai

# TODO: Change PROJECT_ID
PROJECT_ID = "YOUR_PROJECT_ID" # <--- CHANGE THIS
LOCATION = "europe-west4"
MODEL_GOOGLE = "gemini-2.0-flash"


gemini_client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)

def predict(prompt, max_output_tokens, temperature, top_p, top_k):

    response = gemini_client.models.generate_content(
    model=MODEL_GOOGLE, contents=prompt,
        config=GenerateContentConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            candidate_count=1,
            seed=5,
            max_output_tokens=max_output_tokens,
            stop_sequences=["STOP!"],
            presence_penalty=0.0,
            frequency_penalty=0.0,
        ),
    )
    return response.text

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

demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
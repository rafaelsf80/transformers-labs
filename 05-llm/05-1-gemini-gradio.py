""" Simple prompting Gradio demo 
"""

import gradio as gr

from google import genai
from google.genai.types import GenerateContentConfig

MODEL_GOOGLE = "gemini-2.5-flash"

import getpass
google_api_key = getpass.getpass() 

google_client = genai.Client(api_key=google_api_key)
def predict(prompt, max_output_tokens, temperature, top_p, top_k):

    response = google_client.models.generate_content(
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
      gr.Slider(4096, 8192, value=4096, step = 128, label = "max_output_tokens"),
      gr.Slider(0, 1, value=0, step = 0.1, label = "temperature"),
      gr.Slider(1, 5, value=1, step = 1, label = "top_p"),
      gr.Slider(20, 400, value=40, step = 10, label = "top_k"),
    ],
    "text"
    )

demo.launch(server_name="0.0.0.0", server_port=7860, share=True, debug=True)
"""
Gradio Blocks demo — custom layout with rows and columns.
"""

import torch
import gradio as gr
from transformers import pipeline

##############################################################################
# Device selection
##############################################################################
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

generator = pipeline("text-generation", model="distilgpt2", device=device)

def generate(prompt, max_tokens, temperature):
    """Generate text with configurable temperature."""
    results = generator(
        prompt,
        max_new_tokens=int(max_tokens),
        do_sample=True,
        temperature=float(temperature),
    )
    return results[0]["generated_text"]

##############################################################################
# Build a Blocks-based layout
##############################################################################
with gr.Blocks(title="DistilGPT-2 Playground") as demo:

    gr.Markdown("## DistilGPT-2 Playground")
    gr.Markdown("A custom-layout demo using `gr.Blocks`.")

    ##########################################################################
    # Row 1 — input prompt and parameter controls side by side
    ##########################################################################
    with gr.Row():
        with gr.Column(scale=2):
            prompt_box = gr.Textbox(
                label="Prompt",
                placeholder="The meaning of life is...",
                lines=4,
            )
        with gr.Column(scale=1):
            token_slider = gr.Slider(10, 200, value=60, step=10, label="Max Tokens")
            temp_slider = gr.Slider(0.1, 1.5, value=0.7, step=0.1, label="Temperature")
            generate_btn = gr.Button("Generate", variant="primary")

    ##########################################################################
    # Row 2 — output
    ##########################################################################
    output_box = gr.Textbox(label="Output", lines=8)

    ##########################################################################
    # Wire the button click to the generate function
    ##########################################################################
    generate_btn.click(
        fn=generate,
        inputs=[prompt_box, token_slider, temp_slider],
        outputs=output_box,
    )

if __name__ == "__main__":
    demo.launch()
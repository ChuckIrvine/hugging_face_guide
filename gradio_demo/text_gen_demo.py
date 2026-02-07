"""
Gradio demo backed by a Hugging Face text-generation pipeline.
Supports Apple MPS, CUDA, and CPU fallback.
"""

import torch
import gradio as gr
from transformers import pipeline

##############################################################################
# Device selection — prefer Apple MPS, then CUDA, then CPU
##############################################################################
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

print(f"Using device: {device}")

##############################################################################
# Load the text-generation pipeline with a lightweight model
##############################################################################
generator = pipeline(
    "text-generation",
    model="distilgpt2",
    device=device,
)

def generate_text(prompt: str, max_tokens: int) -> str:
    """
    Accept a user prompt and max-token count, return generated text.
    """
    results = generator(
        prompt,
        max_new_tokens=int(max_tokens),
        do_sample=True,
        temperature=0.7,
    )
    # The pipeline returns a list of dicts; extract the first result
    return results[0]["generated_text"]

##############################################################################
# Build the Gradio Interface with multiple input components
##############################################################################
demo = gr.Interface(
    fn=generate_text,
    inputs=[
        gr.Textbox(
            label="Prompt",
            placeholder="Once upon a time...",
            lines=3,
        ),
        gr.Slider(
            minimum=10,
            maximum=200,
            value=50,
            step=10,
            label="Max New Tokens",
        ),
    ],
    outputs=gr.Textbox(label="Generated Text", lines=8),
    title="Text Generation with DistilGPT-2",
    description="Enter a prompt and adjust the token slider to generate text.",
    flagging_mode="never",
)

##############################################################################
# Launch — set share=True to get a temporary public URL
##############################################################################
if __name__ == "__main__":
    demo.launch()
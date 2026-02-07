"""
Minimal Gradio demo — a simple greeting function.
Demonstrates Interface, input/output components, and launch().
"""

import gradio as gr

def greet(name: str) -> str:
    """Return a personalised greeting."""
    return f"Hello, {name}! Welcome to the Gradio demo."

##############################################################################
# Build the Gradio Interface
# - fn: the Python callable
# - inputs: component type(s) for user input
# - outputs: component type(s) for displaying results
##############################################################################
demo = gr.Interface(
    fn=greet,
    inputs=gr.Textbox(label="Your Name"),
    outputs=gr.Textbox(label="Greeting"),
    title="Hello Gradio",
    description="Type your name and press Submit.",
)

##############################################################################
# Launch the server on localhost
##############################################################################
if __name__ == "__main__":
    demo.launch()
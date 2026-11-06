"""Gradio demo. Usage: python app/demo.py --adapter outputs/claimsense-lora"""
import argparse
import sys
from pathlib import Path

import gradio as gr

sys.path.append(str(Path(__file__).parent.parent / "src"))
from inference import generate, load  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--adapter", default="outputs/claimsense-lora")
model, tok = load(ap.parse_args().adapter)

demo = gr.Interface(
    fn=lambda task, text: generate(model, tok, task, text),
    inputs=[gr.Radio(["extract", "coverage", "fraud"], value="extract", label="Task"),
            gr.Textbox(lines=8, label="Claim note / policy clause")],
    outputs=gr.Code(language="json", label="ClaimSense output"),
    title="ClaimSense-LLM",
    description="Insurance claims triage, coverage reasoning and fraud-signal detection. Decision support only.",
)

if __name__ == "__main__":
    demo.launch()

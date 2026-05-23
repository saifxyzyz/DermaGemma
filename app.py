"""Gradio UI for Dermagemma.

Loads the ViT classifier, the RAG retriever, and Gemma at startup so analysis
requests are snappy. Use --skip-llm to disable Gemma entirely (useful on
memory-constrained machines or for quick demos of the classifier + RAG only).

Reports are written to ./reports/ with a timestamped filename.

Run:
    python app.py                # full pipeline
    python app.py --skip-llm     # classifier + RAG only
"""

from __future__ import annotations

import argparse
import base64
import datetime as _dt
import os

import gradio as gr
from PIL import Image

from main import (
    DermagemmaPipeline,
    build_prompt,
    format_rag_context,
    format_vit_predictions,
    generate_pdf,
    map_morphology,
)

REPORTS_DIR = "reports"
os.makedirs(REPORTS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Launch-time configuration
# ---------------------------------------------------------------------------

_parser = argparse.ArgumentParser(description="Dermagemma Gradio UI.")
_parser.add_argument(
    "--skip-llm",
    action="store_true",
    help="Disable Gemma synthesis entirely.",
)
_parser.add_argument("--host", default="127.0.0.1")
_parser.add_argument("--port", type=int, default=7860)
ARGS = _parser.parse_args()
LLM_ENABLED = not ARGS.skip_llm


# ---------------------------------------------------------------------------
# Pipeline (loaded once at import time)
# ---------------------------------------------------------------------------

print("Initializing Dermagemma pipeline...")
PIPELINE = DermagemmaPipeline(load_llm=LLM_ENABLED)
if not LLM_ENABLED:
    print("LLM disabled at launch (--skip-llm).")
print("Ready.")


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _empty_pdf_html(msg: str = "_The PDF report will appear here after analysis._") -> str:
    return (
        '<div style="padding: 60px 20px; text-align: center; color: #64748b; '
        'background: #f8fafc; border: 1px dashed #cbd5e1; border-radius: 8px;">'
        f"{msg}</div>"
    )


def _pdf_to_iframe(pdf_path: str) -> str:
    """Embed a PDF inline using a base64 data URL."""
    with open(pdf_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return (
        f'<iframe src="data:application/pdf;base64,{b64}" '
        f'width="100%" height="780px" '
        f'style="border: 1px solid #e2e8f0; border-radius: 8px;"></iframe>'
    )


def _timestamped_pdf_path() -> str:
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(REPORTS_DIR, f"dermagemma_{ts}.pdf")


# ---------------------------------------------------------------------------
# Inference callback
# ---------------------------------------------------------------------------

def analyze(image: Image.Image, top_k: int, progress=gr.Progress()):
    """Run the full pipeline and return predictions + embedded PDF + download."""
    empty_file = gr.update(value=None, visible=False)

    if image is None:
        return (
            {},
            _empty_pdf_html("_Upload a skin image to begin._"),
            empty_file,
            "Idle. No image provided.",
        )

    progress(0.1, desc="Classifying image with ViT...")
    predictions, morph = PIPELINE.classifier.predict(image, top_k=int(top_k))
    label_scores = {p["condition"]: p["confidence"] for p in predictions}

    progress(0.35, desc="Retrieving knowledge-base entries...")
    blocks = PIPELINE.retriever.retrieve(label_scores)

    if not LLM_ENABLED:
        return (
            label_scores,
            _empty_pdf_html(
                "_LLM disabled at launch. Relaunch without `--skip-llm` to generate reports._"
            ),
            empty_file,
            f"Done. {len(blocks)} KB entries retrieved. LLM disabled at launch.",
        )

    progress(0.6, desc="Generating consultation note with Gemma 4...")
    vit_str = format_vit_predictions(predictions)
    rag_str = format_rag_context(blocks)
    density, distribution = map_morphology(
        morph["mean_activation"], morph["variance_activation"]
    )
    prompt = build_prompt(
        vit_str, density, distribution, morph["mean_activation"], rag_str
    )
    note = PIPELINE.synthesizer.synthesize(prompt)
    note = note.replace("\\%", "%").replace("$", "").strip()

    if not note:
        return (
            label_scores,
            _empty_pdf_html("_Model produced empty output. Try re-running._"),
            empty_file,
            "Done with empty LLM output.",
        )

    progress(0.9, desc="Rendering PDF report...")
    pdf_path = _timestamped_pdf_path()
    try:
        generate_pdf(
            clean_report_text=note,
            auto_labels=vit_str,
            structural_density=density,
            cellular_distribution=distribution,
            activation_score=round(morph["mean_activation"], 4),
            output_filename=pdf_path,
        )
        embed_html = _pdf_to_iframe(pdf_path)
        file_update = gr.update(value=pdf_path, visible=True)
    except Exception as e:
        print(f"PDF generation failed: {e}")
        embed_html = _empty_pdf_html(f"_PDF generation failed: {e}_")
        file_update = empty_file
        pdf_path = None

    status = (
        f"Done. {len(blocks)} KB entries retrieved, report saved to {pdf_path}."
        if pdf_path
        else "Done, but PDF rendering failed."
    )
    return label_scores, embed_html, file_update, status


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

EXAMPLES = [
    ["test_images/acne_vulgaris_black.jpg", 3],
    ["test_images/keloids.jpeg", 3],
    ["test_images/vitiligo_black.jpeg", 3],
    ["test_images/melasma_black_1.webp", 3],
    ["test_images/lichen_planus_black.jpeg", 3],
    ["test_images/post-inflammatory_hyper_1.jpg", 3],
]


CUSTOM_CSS = """
#dermagemma-header {
    border-radius: 12px;
    padding: 24px 28px;
    background: linear-gradient(135deg, #0b0e14 0%, #1e1b4b 100%);
    color: #ffffff;
    margin-bottom: 16px;
}
#dermagemma-header h1 {
    color: #ffffff !important;
    margin: 0 0 6px 0;
    font-size: 28px;
}
#dermagemma-header p {
    margin: 0;
    color: #cbd5e1;
    font-size: 14px;
    line-height: 1.5;
}
.status-box textarea {
    background-color: #f8fafc !important;
    font-family: ui-monospace, SFMono-Regular, monospace;
    font-size: 12px !important;
}
"""


with gr.Blocks(title="Dermagemma — SOC Dermatology AI") as app:
    with gr.Row(elem_id="dermagemma-header"):
        gr.HTML(
            """
            <div>
              <h1>Dermagemma</h1>
              <p>Offline-first dermatology AI for Skin of Color.<br>
              ViT classifier · Hybrid RAG over DermNet · Gemma 4 E2B-IT (local, Q4_K_M).</p>
            </div>
            """
        )

    with gr.Row():
        with gr.Column(scale=1, min_width=340):
            image_in = gr.Image(type="pil", label="Skin image", height=300)
            top_k = gr.Slider(
                minimum=1, maximum=10, value=3, step=1,
                label="Top-K predictions",
                info="Number of candidate diagnoses to consider.",
            )
            submit = gr.Button("Analyze", variant="primary", size="lg")
            status = gr.Textbox(
                label="Status",
                value=(
                    "Ready. Upload an image or pick an example below."
                    if LLM_ENABLED
                    else "Ready (LLM disabled — predictions only)."
                ),
                interactive=False,
                lines=2,
                elem_classes=["status-box"],
            )
            pred_out = gr.Label(label="Top conditions", num_top_classes=5)

        with gr.Column(scale=2):
            pdf_embed = gr.HTML(
                value=_empty_pdf_html(
                    "_The PDF report will appear here after analysis._"
                    if LLM_ENABLED
                    else "_LLM disabled at launch. Relaunch without `--skip-llm` to generate reports._"
                )
            )
            pdf_download = gr.File(
                label="Download PDF report",
                visible=False,
                interactive=False,
            )

    gr.Examples(
        examples=EXAMPLES,
        inputs=[image_in, top_k],
        label="Example images",
        cache_examples=False,
    )

    gr.Markdown(
        "<sub>Dermagemma is a research prototype, not a medical device. "
        "Do not use the output for clinical diagnosis.</sub>"
    )

    submit.click(
        analyze,
        inputs=[image_in, top_k],
        outputs=[pred_out, pdf_embed, pdf_download, status],
    )


if __name__ == "__main__":
    app.launch(
        server_name=ARGS.host,
        server_port=ARGS.port,
        inbrowser=True,
        theme=gr.themes.Soft(),
        css=CUSTOM_CSS,
    )

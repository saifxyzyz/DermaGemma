"""Gradio UI for Dermagemma.

Loads the ViT classifier and the RAG retriever once at startup. Gemma is
loaded lazily on the first request that has the "AI consultation note"
toggle enabled — so users can run classification + RAG without ever paying
the LLM's load cost (and without risking an OOM on CPU).

Run:
    python app.py              # full pipeline; LLM available via UI toggle
    python app.py --skip-llm   # classifier + RAG only, LLM toggle disabled
"""

from __future__ import annotations

import argparse

import gradio as gr
from PIL import Image

from main import (
    DermagemmaPipeline,
    GemmaSynthesizer,
    build_prompt,
    format_rag_context,
    format_vit_predictions,
    map_morphology,
)


# ---------------------------------------------------------------------------
# Launch-time configuration
# ---------------------------------------------------------------------------

_parser = argparse.ArgumentParser(description="Dermagemma Gradio UI.")
_parser.add_argument(
    "--skip-llm",
    action="store_true",
    help="Disable the Gemma synthesis step entirely. The 'Generate AI note' "
    "toggle is rendered disabled in the UI and the LLM is never loaded.",
)
_parser.add_argument("--host", default="127.0.0.1")
_parser.add_argument("--port", type=int, default=7860)
ARGS = _parser.parse_args()
LLM_ENABLED = not ARGS.skip_llm


# ---------------------------------------------------------------------------
# Pipeline (loaded once at import time)
# ---------------------------------------------------------------------------

print("Initializing Dermagemma pipeline (classifier + retriever)...")
PIPELINE = DermagemmaPipeline(load_llm=False)
if not LLM_ENABLED:
    print("LLM disabled at launch (--skip-llm). Gemma will not be loaded.")
print("Ready.")


def _ensure_llm(progress) -> None:
    """Lazy-load Gemma on first use. Raises if it can't fit in memory."""
    if PIPELINE.synthesizer is not None:
        return
    progress(0.5, desc="Loading Gemma 4 E2B (one-time, slow on CPU)...")
    PIPELINE.synthesizer = GemmaSynthesizer()


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def _format_rag_markdown(blocks: list) -> str:
    if not blocks:
        return (
            "_No matching knowledge-base entries retrieved._\n\n"
            "The classifier's top predictions don't match any pathology in the loaded KB. "
            "If you're using the v1 classifier (15 SOC classes) with the full DermNet KB, "
            "this is expected for many predictions — most KB entries belong to conditions "
            "outside v1's vocabulary."
        )
    chunks = []
    for b in blocks:
        soc = "🎯 **SOC-relevant** · " if b.get("soc_relevant") else ""
        heading = b.get("heading") or b["type"].upper()
        url = b.get("source_url") or ""
        source_label = b.get("source", "unknown")
        link = f"[{source_label}]({url})" if url.startswith("http") else source_label
        chunks.append(
            f"### {soc}{heading}\n"
            f"_{b['type'].upper()} · {link}_\n\n"
            f"{b['text']}"
        )
    return "\n\n---\n\n".join(chunks)


def _format_morphology_markdown(morph: dict) -> str:
    density, distribution = map_morphology(
        morph["mean_activation"], morph["variance_activation"]
    )
    return (
        f"- **Structural Tissue Density:** {density}\n"
        f"- **Cellular Architecture:** {distribution}\n"
        f"- **Mean Activation:** {morph['mean_activation']:.4f}\n"
        f"- **Variance Activation:** {morph['variance_activation']:.4f}\n\n"
        "_These are statistical signatures pulled from the ViT's last hidden "
        "layer. They're not clinically validated descriptors — they exist to "
        "give the LLM textual hooks to anchor onto._"
    )


# ---------------------------------------------------------------------------
# Inference callback
# ---------------------------------------------------------------------------

def analyze(image: Image.Image, top_k: int, with_llm: bool, progress=gr.Progress()):
    if image is None:
        return (
            {},
            "_Upload a skin image to begin._",
            "",
            "",
            "Idle — no image provided.",
        )

    progress(0.1, desc="Classifying image...")
    predictions, morph = PIPELINE.classifier.predict(image, top_k=int(top_k))
    label_scores = {p["condition"]: p["confidence"] for p in predictions}

    progress(0.35, desc="Retrieving knowledge-base entries...")
    blocks = PIPELINE.retriever.retrieve(label_scores)

    rag_md = _format_rag_markdown(blocks)
    morph_md = _format_morphology_markdown(morph)

    if not LLM_ENABLED:
        return (
            label_scores,
            rag_md,
            "_LLM disabled at launch (`python app.py --skip-llm`). "
            "Restart without that flag to enable AI consultation notes._",
            morph_md,
            f"Done — {len(blocks)} KB entries retrieved (LLM disabled at launch).",
        )

    if not with_llm:
        return (
            label_scores,
            rag_md,
            "_AI consultation note disabled. Enable the toggle to generate one._",
            morph_md,
            f"Done — {len(blocks)} KB entries retrieved (LLM skipped).",
        )

    _ensure_llm(progress)
    progress(0.7, desc="Generating consultation note with Gemma...")

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

    return (
        label_scores,
        rag_md,
        note or "_Model produced empty output. Try re-running._",
        morph_md,
        f"Done — {len(blocks)} KB entries · LLM note generated.",
    )


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

EXAMPLES = [
    ["test_images/acne_vulgaris_black.jpg", 3, False],
    ["test_images/keloids.jpeg", 3, False],
    ["test_images/vitiligo_black.jpeg", 3, False],
    ["test_images/melasma_black_1.webp", 3, False],
    ["test_images/lichen_planus_black.jpeg", 3, False],
    ["test_images/post-inflammatory_hyper_1.jpg", 3, False],
]


with gr.Blocks(title="Dermagemma — SOC Dermatology AI") as app:
    gr.Markdown(
        """
        # Dermagemma 🩺
        **Closing the diagnostic accuracy gap for Skin of Color (Fitzpatrick IV–VI).**

        Upload a skin image. The pipeline will:
        1. Classify it with a fine-tuned Vision Transformer
        2. Retrieve matching DermNet entries (SOC-prioritized via re-ranking)
        3. Optionally write a structured consultation note with Gemma 4 E2B-IT
        """
    )

    with gr.Row():
        with gr.Column(scale=1, min_width=320):
            image_in = gr.Image(type="pil", label="Skin image", height=320)
            top_k = gr.Slider(
                minimum=1, maximum=10, value=3, step=1, label="Top-K predictions"
            )
            with_llm = gr.Checkbox(
                label=(
                    "Generate AI consultation note (Gemma 4)"
                    if LLM_ENABLED
                    else "Generate AI consultation note (disabled — relaunch without --skip-llm)"
                ),
                info=(
                    "Slow on CPU (1–5 min per request) and memory-heavy. "
                    "Loaded lazily on first use."
                    if LLM_ENABLED
                    else "LLM was disabled at launch via --skip-llm."
                ),
                value=False,
                interactive=LLM_ENABLED,
            )
            submit = gr.Button("Analyze", variant="primary", size="lg")
            status = gr.Textbox(label="Status", value="Idle.", interactive=False)
            gr.Examples(
                examples=EXAMPLES,
                inputs=[image_in, top_k, with_llm],
                label="Example test images",
                cache_examples=False,
            )

        with gr.Column(scale=2):
            with gr.Tab("Predictions"):
                pred_out = gr.Label(label="Top conditions", num_top_classes=5)
                gr.Markdown("### Morphology tokens")
                morph_out = gr.Markdown()
            with gr.Tab("Retrieved Evidence"):
                rag_out = gr.Markdown(
                    value="_Run analysis to see retrieved knowledge-base entries._"
                )
            with gr.Tab("AI Consultation Note"):
                note_out = gr.Markdown(
                    value="_Run analysis with the AI toggle on to see a generated note._"
                )

    submit.click(
        analyze,
        inputs=[image_in, top_k, with_llm],
        outputs=[pred_out, rag_out, note_out, morph_out, status],
    )


if __name__ == "__main__":
    app.launch(
        server_name=ARGS.host,
        server_port=ARGS.port,
        inbrowser=True,
        theme=gr.themes.Soft(),
    )

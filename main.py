"""Dermagemma end-to-end inference pipeline.

Combines the local ViT skin classifier (from vit.py) with the RAG retrieval
engine, Gemma 4 synthesis, and PDF export from the Colab notebook. Defaults
are tuned for CPU; if a CUDA device is present it will be used automatically.

Usage:
    python main.py path/to/image.jpg
    python main.py path/to/image.jpg --pdf report.pdf
    python main.py path/to/image.jpg --skip-llm        # classification + RAG only
"""

import argparse
import datetime as _dt
import hashlib
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float16 if DEVICE.type == "cuda" else torch.float32

VIT_PATH = "saif0z/vit_skin_classifier"
EMBEDDER_PATH = "pritamdeka/S-PubMedBert-MS-MARCO"
GEMMA_GGUF_PATH = "models/gemma-4-E2B-it-Q4_K_M.gguf"
KB_PATH = "knowledge_base.json"
EMBED_CACHE_PATH = ".kb_embeddings.npy"
EMBED_HASH_PATH = ".kb_embeddings.hash"


# One canonical sample per disease present in test_images/.
# Names match the v1 classifier's class labels where applicable; the bottom
# block lists samples whose ground-truth condition is OUTSIDE the v1 vocab
# (Melanoma, Squamous Cell Carcinoma, Urticaria, Scabies) — useful for
# observing how the model behaves on out-of-distribution input.
TEST_SUITE = [
    ("Acne Vulgaris", "test_images/acne_vulgaris_black.jpg"),
    ("Keloids", "test_images/keloids.jpeg"),
    ("Lichen Planus", "test_images/lichen_planus_black.jpeg"),
    ("Melasma", "test_images/melasma_black_1.webp"),
    ("Post-Inflammatory Hyperpigmentation", "test_images/post-inflammatory_hyper_1.jpg"),
    ("Vitiligo", "test_images/vitiligo_black.jpeg"),
    ("Melanoma", "test_images/melanoma_1.jpg"),
    ("Squamous Cell Carcinoma", "test_images/squamous-cell-carcinoma-on-leg-of-black-person.jpg"),
    ("Urticaria", "test_images/urticaria.jpg"),
    ("Scabies", "test_images/VWH-Wikicommons-Scabies-infection-01-996dfd1ad459458a8b173b341aaa4643.jpg"),
]


def load_knowledge_base(path: str = KB_PATH) -> List[dict]:
    """Load knowledge base entries from disk.

    Supports both legacy flat-list JSON and the {_meta, entries} payload
    produced by src/scrape_dermnet.py.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Knowledge base not found at {path}. "
            f"Generate it with: python src/scrape_dermnet.py"
        )
    with open(path) as f:
        data = json.load(f)
    entries = data["entries"] if isinstance(data, dict) and "entries" in data else data
    soc = sum(1 for e in entries if e.get("soc_relevant"))
    print(f"Loaded {len(entries)} KB entries from {path} ({soc} SOC-tagged).")
    return entries


# ---------------------------------------------------------------------------
# 1. ViT classifier — drives both top-k predictions and morphology features
# ---------------------------------------------------------------------------

class SkinClassifier:
    def __init__(self, model_path: str = VIT_PATH, device: torch.device = DEVICE):
        print(f"Loading ViT classifier from {model_path} on {device}...")
        self.processor = ViTImageProcessor.from_pretrained(model_path)
        self.model = ViTForImageClassification.from_pretrained(model_path).to(device)
        self.model.eval()
        self.device = device
        self.id2label = self.model.config.id2label
        print(f"ViT loaded — {len(self.id2label)} conditions.")

    @torch.no_grad()
    def predict(self, image, top_k: int = 3) -> Tuple[List[dict], Dict[str, float]]:
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        else:
            image = image.convert("RGB")

        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs, output_hidden_states=True)

        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
        top_probs, top_indices = probs.topk(top_k)
        predictions = [
            {
                "condition": self.id2label[idx.item()].replace("_", " "),
                "confidence": prob.item(),
            }
            for prob, idx in zip(top_probs, top_indices)
        ]

        # Morphology tokens extracted from the final hidden state, mirroring
        # the notebook's "structural density" / "cellular distribution" heuristics.
        last_hidden = outputs.hidden_states[-1]  # [1, 197, 768]
        morphology = {
            "mean_activation": float(last_hidden.mean().cpu()),
            "variance_activation": float(last_hidden.var().cpu()),
        }
        return predictions, morphology


def map_morphology(mean_act: float, var_act: float) -> Tuple[str, str]:
    structural_density = (
        "Marked Fibrotic / Hyper-keratotic" if mean_act > 0
        else "Moderate Epidermal Overgrowth"
    )
    cellular_distribution = (
        "High Asymmetric Irregularity" if var_act > 1.0
        else "Homogeneous Dermal Expansion"
    )
    return structural_density, cellular_distribution


# ---------------------------------------------------------------------------
# 2. RAG retriever — hybrid BM25 + dense embeddings with RRF
# ---------------------------------------------------------------------------

class DermatologyRetriever:
    def __init__(
        self,
        knowledge_base: List[dict],
        embedder_path: str = EMBEDDER_PATH,
        label_threshold: float = 0.3,
        top_n_total: int = 3,
        rrf_k: int = 60,
        soc_boost: float = 0.20,
    ):
        from rank_bm25 import BM25Okapi
        from sentence_transformers import SentenceTransformer
        import faiss

        print(f"Loading sentence embedder ({embedder_path})...")
        self.embedder = SentenceTransformer(embedder_path, device=DEVICE.type)
        self.kb = knowledge_base
        self.label_threshold = label_threshold
        self.top_n_total = top_n_total
        self.rrf_k = rrf_k
        self.soc_boost = soc_boost

        corpus_texts = [s["text"] for s in knowledge_base]
        self.bm25 = BM25Okapi([self._tok(t) for t in corpus_texts])
        embeddings = self._load_or_compute_embeddings(corpus_texts)
        self.faiss_index = faiss.IndexFlatIP(embeddings.shape[1])
        self.faiss_index.add(embeddings)

    def _load_or_compute_embeddings(self, corpus_texts: List[str]) -> np.ndarray:
        """Cache embeddings to disk keyed by a hash of the corpus contents."""
        h = hashlib.sha1()
        for t in corpus_texts:
            h.update(t.encode("utf-8"))
            h.update(b"\0")
        corpus_hash = h.hexdigest()

        if os.path.exists(EMBED_CACHE_PATH) and os.path.exists(EMBED_HASH_PATH):
            with open(EMBED_HASH_PATH) as f:
                if f.read().strip() == corpus_hash:
                    arr = np.load(EMBED_CACHE_PATH)
                    print(f"Loaded cached embeddings from {EMBED_CACHE_PATH} ({arr.shape}).")
                    return arr

        print(f"Encoding {len(corpus_texts)} KB entries (one-time cost)...")
        arr = self.embedder.encode(corpus_texts, normalize_embeddings=True).astype("float32")
        np.save(EMBED_CACHE_PATH, arr)
        with open(EMBED_HASH_PATH, "w") as f:
            f.write(corpus_hash)
        print(f"Cached embeddings to {EMBED_CACHE_PATH}.")
        return arr

    @staticmethod
    def _tok(text: str) -> List[str]:
        return text.lower().split()

    def retrieve(self, predicted_labels: Dict[str, float]) -> List[dict]:
        active = [(l, p) for l, p in predicted_labels.items() if p >= self.label_threshold]
        active.sort(key=lambda x: -x[1])

        if not active:
            return [s for s in self.kb if s["pathology"] == "Normal"]

        active_labels = {l.lower().replace("_", " ") for l, _ in active}
        label_to_confidence = {l.lower().replace("_", " "): p for l, p in active}

        candidate_idxs = [
            i for i, s in enumerate(self.kb)
            if s["pathology"].lower().replace("_", " ") in active_labels
        ]
        if not candidate_idxs:
            return []

        query = " ".join(l for l, _ in active) + " skin of color presentation features guidelines"

        emb = self.embedder.encode([query], normalize_embeddings=True).astype("float32")
        _, dense_ix = self.faiss_index.search(emb, len(self.kb))
        dense_order = dense_ix[0].tolist()

        bm25_scores = self.bm25.get_scores(self._tok(query))
        bm25_order = np.argsort(bm25_scores)[::-1].tolist()

        scored = []
        for i in candidate_idxs:
            rrf = 0.0
            if i in dense_order:
                rrf += 1.0 / (self.rrf_k + dense_order.index(i) + 1)
            if i in bm25_order:
                rrf += 1.0 / (self.rrf_k + bm25_order.index(i) + 1)

            pathology_name = self.kb[i]["pathology"].lower().replace("_", " ")
            rrf *= (1.0 + label_to_confidence.get(pathology_name, 1.0))

            type_boost = {
                "phrasing": 0.15, "feature": 0.12, "pitfall": 0.10,
                "treatment": 0.08, "definition": 0.05,
            }
            rrf += type_boost.get(self.kb[i]["type"], 0.0)

            # SOC-tagged entries are the project's whole point — boost them.
            if self.kb[i].get("soc_relevant"):
                rrf += self.soc_boost

            scored.append((i, rrf))

        scored.sort(key=lambda x: -x[1])
        return [self.kb[i] for i, _ in scored[: self.top_n_total]]


# ---------------------------------------------------------------------------
# 3. Prompt + Gemma synthesis
# ---------------------------------------------------------------------------

def format_vit_predictions(predictions: List[dict]) -> str:
    return ", ".join(
        f"{p['condition']} (Confidence: {p['confidence'] * 100:.0f}%)"
        for p in predictions
    )


def format_rag_context(blocks: List[dict]) -> str:
    if not blocks:
        return "No specific evidence retrieved from the local knowledge base."
    lines = []
    for b in blocks:
        soc_tag = " | SOC" if b.get("soc_relevant") else ""
        src = b.get("source_url") or b.get("source", "unknown")
        heading = b.get("heading")
        head_str = f" — {heading}" if heading else ""
        lines.append(f"[{b['type'].upper()}{soc_tag} | {src}{head_str}]\n{b['text']}")
    return "\n\n".join(lines)


def build_prompt(vit_str: str, density: str, distribution: str, mean_act: float, rag_str: str) -> str:
    return f"""<start_of_turn>user
You are an expert clinical dermatologist specializing in Skin of Color. Your task is to evaluate a patient case using three clean inputs: diagnostic data, visual morphology tokens, and authoritative clinical guidelines.

[INPUT 1: EMBEDDED VISUAL MORPHOLOGY]
- Structural Tissue Density Profile: {density}
- Cellular Architecture Distribution: {distribution}
- Core Matrix Layer Activation Score: {round(mean_act, 4)}

[INPUT 2: IMAGE CLASSIFIER ESTIMATES]
{vit_str}

[INPUT 3: AUTHORITATIVE CLINICAL EVIDENCE]
{rag_str}

INSTRUCTIONS:
Synthesize the inputs above to generate a formal, authoritative Dermatology Consultation Note. Do not repeat these instructions or raw inputs. Output the report immediately starting with the headers below:

1. CLINICAL EXAMINATION & FINDINGS
2. IMPRESSION & MANAGEMENT PLAN<end_of_turn>
<start_of_turn>model
"""


class GemmaSynthesizer:
    """Local Gemma 4 E2B-IT via llama-cpp-python (Q4_K_M GGUF). CPU-friendly."""

    def __init__(self, model_path: str = GEMMA_GGUF_PATH, n_ctx: int = 4096, n_threads: int = None):
        from llama_cpp import Llama

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"GGUF not found at {model_path}. Download it with:\n"
                f"  hf download unsloth/gemma-4-E2B-it-GGUF gemma-4-E2B-it-Q4_K_M.gguf "
                f"--local-dir models/"
            )

        threads = n_threads or os.cpu_count() or 4
        print(
            f"Loading Gemma GGUF ({model_path}) — n_ctx={n_ctx}, n_threads={threads}. "
            f"This takes 10-30s on CPU; progress below.",
            flush=True,
        )
        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=threads,
            verbose=True,
        )
        print("Gemma ready.", flush=True)

    def synthesize(self, prompt: str, max_new_tokens: int = 300) -> str:
        out = self.llm.create_completion(
            prompt=prompt,
            max_tokens=max_new_tokens,
            temperature=0.2,
            repeat_penalty=1.2,
            stop=["<end_of_turn>", "<start_of_turn>"],
        )
        return out["choices"][0]["text"]


# ---------------------------------------------------------------------------
# 4. PDF export
# ---------------------------------------------------------------------------

def generate_pdf(
    clean_report_text: str,
    auto_labels: str,
    structural_density: str,
    cellular_distribution: str,
    activation_score: float,
    output_filename: str,
) -> str:
    from weasyprint import HTML

    formatted_body = clean_report_text
    formatted_body = formatted_body.replace(
        "1. CLINICAL EXAMINATION & FINDINGS",
        '<div class="section-title">1. Clinical Examination &amp; Findings</div>',
    )
    formatted_body = formatted_body.replace(
        "2. IMPRESSION & MANAGEMENT PLAN",
        '<div class="section-title">2. Impression &amp; Management Plan</div>',
    )
    formatted_body = formatted_body.replace("**Impression:**", "<strong>Impression:</strong>")
    formatted_body = formatted_body.replace("**Management Plan:**", "<strong>Management Plan:</strong>")
    formatted_body = formatted_body.replace("\n", "<br>")

    eval_date = _dt.date.today().strftime("%B %d, %Y")

    html_template = f"""
    <!DOCTYPE html>
    <html><head><meta charset="UTF-8"><style>
        @page {{ size: A4; margin: 15mm 12mm; background-color: #ffffff; }}
        body {{ font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; color: #2c3e50; margin: 0; padding: 0; font-size: 11pt; line-height: 1.5; }}
        .header-banner {{ background-color: #0b0e14; color: #ffffff; margin: -15mm -12mm 20px -12mm; padding: 25px 12mm; border-bottom: 5px solid #4f46e5; }}
        .header-banner h1 {{ margin: 0; font-size: 22pt; font-weight: bold; letter-spacing: 0.5px; }}
        .header-banner p {{ margin: 5px 0 0 0; font-size: 11pt; color: #94a3b8; }}
        .meta-grid {{ width: 100%; border-collapse: collapse; margin-bottom: 25px; }}
        .meta-grid td {{ padding: 6px 10px; border: 1px solid #e2e8f0; font-size: 10pt; }}
        .meta-label {{ background-color: #f8fafc; font-weight: bold; color: #1e1b4b; width: 25%; }}
        .section-box {{ background-color: #f1f5f9; border-left: 4px solid #0891b2; padding: 12px 15px; margin-bottom: 25px; border-radius: 0 6px 6px 0; }}
        .section-box h3 {{ margin: 0 0 8px 0; color: #0891b2; font-size: 12pt; text-transform: uppercase; }}
        .section-box ul {{ margin: 0; padding-left: 20px; font-size: 10pt; }}
        .section-title {{ font-size: 13pt; font-weight: bold; color: #4f46e5; margin-top: 25px; margin-bottom: 10px; border-bottom: 1px solid #e2e8f0; padding-bottom: 4px; page-break-inside: avoid; page-break-after: avoid; }}
        .report-text {{ font-size: 11pt; text-align: justify; color: #334155; }}
        .footer {{ position: fixed; bottom: 0; left: 0; right: 0; text-align: center; font-size: 8pt; color: #94a3b8; border-top: 1px solid #e2e8f0; padding-top: 5px; }}
    </style></head><body>
        <div class="header-banner">
            <h1>DERMAGEMMA CONSULTATION NOTE</h1>
            <p>Multimodal AI Dermatological Evaluation Backend System</p>
        </div>
        <table class="meta-grid">
            <tr>
                <td class="meta-label">Document Type</td><td>Automated Consultation Chart</td>
                <td class="meta-label">Evaluation Date</td><td>{eval_date}</td>
            </tr>
            <tr>
                <td class="meta-label">Primary Classifier Path</td>
                <td colspan="3" style="font-weight: bold; color: #4f46e5;">{auto_labels}</td>
            </tr>
        </table>
        <div class="section-box">
            <h3>Extracted Layer Morphology Tokens</h3>
            <ul>
                <li><strong>Structural Tissue Density Profile:</strong> {structural_density}</li>
                <li><strong>Cellular Architecture Distribution:</strong> {cellular_distribution}</li>
                <li><strong>Core Matrix Layer Activation Score:</strong> {activation_score}</li>
            </ul>
        </div>
        <div class="report-text">{formatted_body}</div>
        <div class="footer">Dermagemma Framework Evaluation Report • Confidential Medical Informatics System</div>
    </body></html>
    """

    temp_html_path = "_dermagemma_report.html"
    with open(temp_html_path, "w", encoding="utf-8") as f:
        f.write(html_template)
    HTML(temp_html_path).write_pdf(output_filename)
    os.remove(temp_html_path)
    print(f"PDF report saved to '{output_filename}'.")
    return output_filename


# ---------------------------------------------------------------------------
# 5. End-to-end pipeline (reusable)
# ---------------------------------------------------------------------------

class DermagemmaPipeline:
    """Loads ViT + retriever (+ optional Gemma) once; reuse via .analyze()."""

    def __init__(self, load_llm: bool = True, kb_path: str = KB_PATH):
        self.classifier = SkinClassifier()
        kb = load_knowledge_base(kb_path)
        self.retriever = DermatologyRetriever(kb)
        self.synthesizer = GemmaSynthesizer() if load_llm else None

    def analyze(self, image_path: str, top_k: int = 3, output_pdf: str = None) -> dict:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        predictions, morph = self.classifier.predict(image_path, top_k=top_k)
        print(f"\n[{image_path}] Top predictions:")
        for p in predictions:
            print(f"  - {p['condition']}: {p['confidence'] * 100:.2f}%")

        predicted_labels = {p["condition"]: p["confidence"] for p in predictions}
        blocks = self.retriever.retrieve(predicted_labels)
        print(f"Retrieved {len(blocks)} knowledge-base entries.")

        vit_str = format_vit_predictions(predictions)
        rag_str = format_rag_context(blocks)
        density, distribution = map_morphology(
            morph["mean_activation"], morph["variance_activation"]
        )
        activation_score = round(morph["mean_activation"], 4)

        print("\n--- Retrieved Evidence ---")
        print(rag_str)
        print("--------------------------\n")

        result = {
            "image": image_path,
            "predictions": predictions,
            "morphology": morph,
            "retrieved": blocks,
            "note": None,
            "pdf": None,
        }

        if self.synthesizer is None:
            print("LLM disabled; skipping Gemma synthesis.")
            return result

        prompt = build_prompt(
            vit_str, density, distribution, morph["mean_activation"], rag_str
        )
        note = self.synthesizer.synthesize(prompt)
        clean_note = note.replace("\\%", "%").replace("$", "")
        result["note"] = clean_note

        print(
            "\n========================================================================\n"
            "                 DERMAGEMMA DIGITAL CLINICAL CHARTS\n"
            "========================================================================\n"
            f"{clean_note.strip()}\n"
            "========================================================================\n"
        )

        if output_pdf:
            stripped = (
                clean_note.replace("=" * 72, "")
                .replace("DERMAGEMMA DIGITAL CLINICAL CHARTS", "")
                .strip()
            )
            generate_pdf(
                clean_report_text=stripped,
                auto_labels=vit_str,
                structural_density=density,
                cellular_distribution=distribution,
                activation_score=activation_score,
                output_filename=output_pdf,
            )
            result["pdf"] = output_pdf

        return result


def _norm_label(name: str) -> str:
    """Normalize label for comparison: collapse separators, strip trailing 's'."""
    n = name.lower().replace("-", " ").replace("_", " ").strip()
    if n.endswith("s") and len(n) > 3:
        n = n[:-1]
    return n


def run_test_suite(pipeline: "DermagemmaPipeline", top_k: int) -> None:
    """Run the built-in one-image-per-disease test set and print a summary."""
    known_norms = {_norm_label(v) for v in pipeline.classifier.id2label.values()}
    rows = []
    for expected, image_path in TEST_SUITE:
        print("\n" + "=" * 72)
        print(f"EXPECTED: {expected}    IMAGE: {image_path}")
        print("=" * 72)
        try:
            result = pipeline.analyze(image_path, top_k=top_k)
            top = result["predictions"][0]
            expected_norm = _norm_label(expected)
            in_vocab = expected_norm in known_norms
            correct = in_vocab and _norm_label(top["condition"]) == expected_norm
            rows.append({
                "expected": expected,
                "predicted": top["condition"],
                "confidence": top["confidence"],
                "in_vocab": in_vocab,
                "correct": correct,
            })
        except FileNotFoundError as e:
            print(f"  SKIPPED: {e}")
            rows.append({"expected": expected, "predicted": "<missing image>",
                         "confidence": 0.0, "in_vocab": False, "correct": False})

    print("\n\n" + "=" * 78)
    print("TEST SUITE SUMMARY")
    print("=" * 78)
    print(f"{'EXPECTED':<38} {'TOP PREDICTION':<32} {'CONF':>6}  STATUS")
    print("-" * 78)
    for r in rows:
        if r["predicted"] == "<missing image>":
            status = "MISSING"
        elif not r["in_vocab"]:
            status = "OOD"
        elif r["correct"]:
            status = "OK"
        else:
            status = "WRONG"
        print(f"{r['expected']:<38} {r['predicted']:<32} {r['confidence'] * 100:>5.1f}%  {status}")

    in_vocab = [r for r in rows if r["in_vocab"]]
    if in_vocab:
        correct = sum(r["correct"] for r in in_vocab)
        print(f"\nIn-vocab accuracy: {correct}/{len(in_vocab)} = {correct / len(in_vocab) * 100:.1f}%")
    ood_count = sum(1 for r in rows if not r["in_vocab"] and r["predicted"] != "<missing image>")
    if ood_count:
        print(f"OOD samples (outside v1 classifier vocab): {ood_count}")


def main():
    parser = argparse.ArgumentParser(description="Dermagemma end-to-end inference pipeline.")
    parser.add_argument("images", nargs="*", help="Path(s) to input skin image(s). Omit when using --test-all.")
    parser.add_argument("--test-all", action="store_true",
                        help="Run the built-in one-image-per-disease test suite from test_images/.")
    parser.add_argument(
        "--pdf",
        default=None,
        help="Output PDF path. With multiple images, used as a prefix (foo.pdf → foo_1.pdf, foo_2.pdf, ...). Ignored with --test-all.",
    )
    parser.add_argument("--top-k", type=int, default=3, help="Number of top classifier predictions to keep.")
    parser.add_argument("--skip-llm", action="store_true", help="Run classification + RAG only; skip Gemma synthesis.")
    args = parser.parse_args()

    if args.test_all and args.images:
        parser.error("Pass either image paths OR --test-all, not both.")
    if not args.test_all and not args.images:
        parser.error("Provide one or more image paths, or use --test-all.")

    pipeline = DermagemmaPipeline(load_llm=not args.skip_llm)

    if args.test_all:
        run_test_suite(pipeline, top_k=args.top_k)
        return

    for i, image in enumerate(args.images, start=1):
        pdf_path = None
        if args.pdf:
            if len(args.images) == 1:
                pdf_path = args.pdf
            else:
                stem, ext = os.path.splitext(args.pdf)
                pdf_path = f"{stem}_{i}{ext or '.pdf'}"
        pipeline.analyze(image, top_k=args.top_k, output_pdf=pdf_path)


if __name__ == "__main__":
    main()



# Dermagemma 🩺✨
### Closing the Healthcare Gap in AI Dermatology for Skin of Color (SOC)

Dermagemma is a dynamic, multimodal Retrieval-Augmented Generation (RAG) assistant designed to address the critical 17% drop in standard skin AI diagnostic accuracy for patients with darker skin tones. By combining Saif's custom-tuned Vision Transformer (ViT) with Google's advanced **Gemma 4 E4B-IT** model, Dermagemma extracts deep visual matrix features, cross-references them with an authoritative skin-of-color knowledge base, and outputs precise, unbiased, and downloadable clinical charts.

---

## 📖 Table of Contents
1. [Project Purpose](#-project-purpose)
2. [How it Works](#%EF%B8%8F-how-it-works)
3. [Quick Start Guide (Google Colab)](#-quick-start-guide-google-colab)
4. [How to Run the Pipeline](#-how-to-run-the-pipeline)
5. [Inference & PDF Generation](#-inference--pdf-generation)

---

## 🎯 Project Purpose
Modern dermatological AI tools are trained predominantly on lighter skin types (Fitzpatrick Scales I-III). This creates a dangerous systemic bias: conditions like eczema, keloids, or hyperpigmentation manifest differently on darker skin types (Fitzpatrick Scales IV-VI), frequently leading to misdiagnosis, wrong prescriptions, and prolonged patient suffering. 

**Dermagemma eliminates this bias** by providing an explainable, end-to-end multimodal pipeline that reads clinical tissue textures accurately, anchors the findings to verified Skin of Color Society data guidelines, and compiles automated expert chart notes.

## ViT Classifier Training Results

| Train Loss | Val Loss | Accuracy | F1 | Precision | Recall |
|---|---|---|---|---|---|
| **0.041** | **0.688** | **82.2%** | **82.0%** | **83.0%** | **82.2%** |
---

## 🛠️ How it Works
1. **Visual Embedding Core:** The system reads an incoming patient skin image through **Saif’s ViT layers**, extracting statistical morphology tokens (tissue density profile, cellular architecture distribution, and matrix activation scores).
2. **Knowledge-Gated Query RAG:** The model utilizes these feature matrices to dynamically isolate clinical conditions and filter a targeted local dictionary database (`KNOWLEDGE_BASE`) covering 15 critical skin-of-color pathologies.
3. **Advanced LLM Synthesis:** The retrieved text guidelines and visual tokens are securely packaged into custom chat wrappers and executed via **Gemma 4 E4B** on a T4 GPU using strict repetition controls.
4. **Automated Vector PDF Export:** The system outputs an on-screen consultation note and instantly compiles a beautifully formatted, downloadable corporate medical PDF chart.

---

## 🚀 Quick Start Guide (Google Colab)
The entire project is structured to run inside a single, self-contained interactive notebook: **`gemma4good_RAG_pipeline.ipynb`**.

### Prerequisites
* A Google Colab account (Colab Pro with High-RAM is recommended, but standard T4 runtimes are supported).
* A Hugging Face account with an active access token (`HF_TOKEN`) authorized to access the gated Google Gemma 4 repositories.
* Your custom user token saved securely in Colab's Secrets manager tab (the key icon 🔑) under the variable name `HF_TOKEN`.

---

## 💻 How to Run the Pipeline

> ⚠️ **CRITICAL EXECUTION NOTE:** Every cell in the master notebook must be run in exact sequential order from top to bottom to maintain the local PyTorch tensor graph configurations.

### Step 1: Initial Hardware Setup & Installation
1. Open **`gemma4good_RAG_pipeline.ipynb`** in Google Colab.
2. Ensure your active runtime environment type is explicitly configured to utilize the **T4 GPU** accelerator hardware block.
3. Run the initial dependency installation cells (`!pip install transformers torch weasyprint ...`).

### Step 2: The Required Cache Restart 🔄
1. Because the dynamic notebook updates deep CUDA and library packages into your instance runtime environment cache, **Colab may prompt you with a message window requesting a Session Restart.**
2. Click **Restart Session** (or navigate manually to the top toolbar menu and select `Runtime` ➡️ `Restart session`).
3. **CRITICAL WARNING:** Once the runtime session boots back up cleanly, **re-run every single cell down the notebook sequentially EXCEPT the initial installation code cells.** This ensures the active memory maps are populated correctly without overwriting your workspace.

### Step 3: Model Ingestion & Weight Consolidation
Run the model configuration cells. The notebook is hard-coded with a custom `{"" : 0}` device layout template configuration map to explicitly pin Gemma 4 E4B parameters directly into your physical T4 GPU hardware card VRAM, completely bypassing virtual `meta` device allocation loop bugs.

---

## 📸 Inference & PDF Generation

To run a live, real-world evaluation diagnostic routine on any specific sample patient case, follow these exact quick steps:

1. **Upload your test image:** Locate a random skin-of-color disease sample image file (such as an example keloid, acne, or PIH image patch). Upload it into your local Colab directory filesystem via the left folder navigation panel (📂). Name the file `test_image.png` and save it directly under the `/content/` path matrix root block (`/content/test_image.png`).
2. **Configure your inference node:** Navigate down to the execution interface cell area block at the very bottom of the notebook file and locate the `IMAGE_PATH` parameter assignment string option marker:
   ```python
   IMAGE_PATH = "/content/test_image.png"

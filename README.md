<!-- PROJECT TITLE -->
# Voice ↔ Description Alignment (ANLP Project)

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/Gal-bloch/ANLP)

Multimodal project aligning short speech segments with natural‑language voice descriptions using (1) a supervised Voice→Text embedding model and (2) Deep Canonical Correlation Analysis (DCCA) variants trained on paired speech–description data (with generated hard negatives).

---

## ✨ Key Capabilities
* Build enriched speech–description datasets (refined + negated descriptions).
* Generate speaker & text embeddings (Resemblyzer + IBM Granite embeddings).
* Train two model families:
  * Voice2Embedding (Transformer encoder → text embedding space).
  * DCCA / DCCAV2 / DCCAV3 (shared latent space via correlation maximization + negatives).
* Text ↔ Voice retrieval (cosine similarity in shared/target space).
* Streamlit interactive demo.
* Evaluation scripts (retrieval metrics, classifier sanity checks, human evaluation harness).

---

## 🧱 Repository Structure (simplified)
```
code/
  create_dataset.py          # Dataset building & enrichment (embeddings, refined + negated descriptions)
  Voice2Embedding.py         # Supervised voice→text model
  DCCA.py                    # Baseline deep CCA model
  DCCAV2.py / DCCAV3.py      # Architectural / training refinements
  demo.py                    # Streamlit retrieval demo
  RetrievalEvaluation.py     # Retrieval evaluation metrics
  ClassifierEvaluation.py    # Auxiliary attribute/quality evaluation
  human_eval.py              # Human evaluation orchestration
  debug_similarity_tool.py   # Embedding similarity inspection
models/                      # Saved checkpoints (.pt)
audio_cache/                 # Temporary audio (demo / preprocessing)
datasets/                    # HuggingFace dataset on disk (train/test)
requirements.txt             # Dependencies
```

---

## 🧪 Modalities & Embeddings
| Modality | Embedder | Dim | Notes |
|----------|----------|-----|-------|
| Speech | Resemblyzer VoiceEncoder | 256 | Raw speech embedding backbone |
| Text | ibm-granite/granite-embedding-125m-english | 768 | SentenceTransformer wrapper |
| Negated Text | Generated via prompt | 768 | Hard negatives (attribute inversion) |

---

## 🧬 Model Families
### Voice2Embedding
Resemblyzer → TransformerEncoder → Projection → (L2-ish normalization). Optimizes cosine similarity (optionally contrastive with negations).

### DCCA (+ V2 / V3)
Two deep MLP encoders (audio/text) map to shared latent (default 128‑d). Objective: maximize canonical correlation between matched pairs; contrastive variant reduces correlation with negated descriptions.

Conceptual objective:
```
maximize Corr( f_audio(a), f_text(t_pos) ) - Corr( f_audio(a), f_text(t_neg) )
```

---

## 📦 Dataset Pipeline (create_dataset.py)
1. Load base CSV metadata & labels (extract csv files from datasets/SPEECHCRAFT_GIGASPEECH_CSVs.zip)
2. Optional refinement: prompt constrains description to labeled vocal attributes.
3. Negated description generation (attribute inversion) for hard negatives.
4. Compute embeddings:
   * Speaker (Resemblyzer)
   * Text (Granite) for original/refined/negated descriptions
5. Persist HuggingFace `DatasetDict` (`ENRICHED_DATASET_V2_PATH`).

Sample columns:
```
segment_id, audio, text_description, negated_description,
resemblyzer_speaker_embedding,
granite_description_embedding, granite_negated_description_embedding,
gender, age, speed, pitch, energy, emotion
```

Security note: Remove the hard‑coded HuggingFace token from `create_dataset.py`; source it via environment variable (`HF_TOKEN`).

---

## 🚀 Quick Start
### 1. Environment
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
Minimal demo dependencies:
```bash
pip install torch torchaudio sentence-transformers datasets streamlit resemblyzer soundfile librosa
```

### 2. Dataset
If `datasets/Enriched_VISTA_dataset_v2` exists, skip. Otherwise adapt (script currently constant‑driven):
```bash
python code/create_dataset.py
```
(Refactor to argparse for sizes / toggles as needed.)

### 3. Train
Voice2Embedding:
```bash
python code/Voice2Embedding.py
```
DCCA:
```bash
python code/DCCA.py
```
(Check constants at top of each script for paths / hyperparams.)

Additional variants:

DCCAV2 (transformer encoders + correlation contrastive w/ negatives, stability tweaks):
```bash
python code/DCCAV2.py
```
Key diffs vs baseline:
* Gradient clipping hooks + explicit norm clamp
* Mean pooling after shallow Transformer (3 layers) per modality
* Contrastive correlation loss (positive − negative)
* Smaller LR (1e-5) & weight decay 1e-4

DCCAV3 (deeper encoders + symmetric InfoNCE in a larger shared space):
```bash
python code/DCCAV3.py
```
Highlights:
* Deeper modality-specific Transformers (5 layers) with LayerNorm + GELU heads
* Larger shared dim (256) and InfoNCE temperature=0.07
* AdamW + CosineAnnealingLR scheduler

Tip: Adjust batch size (env GPU memory), `SHARED_DIM`, and learning rate constants at the top of the script before launching. All scripts save best checkpoints into `models/`.

### 4. Run Demo
```bash
streamlit run code/demo.py
```
Open the local URL → choose checkpoint → text or audio query → retrieve matches.

### 5. Evaluate
```bash
python code/RetrievalEvaluation.py
python code/ClassifierEvaluation.py
python code/human_eval.py
```

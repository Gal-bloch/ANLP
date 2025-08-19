import streamlit as st
import torch
import torchaudio
import numpy as np
import random
from sentence_transformers import SentenceTransformer
from resemblyzer import VoiceEncoder, wav_to_mel_spectrogram
import torch.nn.functional as F
import soundfile as sf
import os
import logging
from datasets import load_from_disk, Audio
from Voice2Embedding import Voice2Embedding, VOICE2EMBEDDING_DESCRIPTION_EMBEDDER
from DCCA import create_dcca_model, DCCA_DESCRIPTION_EMBEDDER, DCCASpeechText
from create_dataset import (
    ENRICHED_DATASET_V2_PATH, 
    AUDIO_COLUMN, 
    DESCRIPTION_COLUMN, 
    GRANITE_DESCRIPTION_EMBEDDING_COLUMN,
    RESEMBLYZER_SPEAKER_EMBEDDING_COLUMN,
    ID_COLUMN
)

# Optional imports for V2/V3 models
try:
    from DCCAV3 import create_dcca_v3_model
    HAS_DCCAV3 = True
except ImportError:
    HAS_DCCAV3 = False
try:
    from DCCAV2 import create_dcca_v2_model
    HAS_DCCAV2 = True
except ImportError:
    HAS_DCCAV2 = False

# Initialize session state
if 'selected_idx' not in st.session_state:
    st.session_state.selected_idx = 0
if 'text_input' not in st.session_state:
    st.session_state.text_input = ""
if 'similarity_result' not in st.session_state:
    st.session_state.similarity_result = None
if 'selected_model_file' not in st.session_state:
    st.session_state.selected_model_file = None
if 'current_model_type' not in st.session_state:
    st.session_state.current_model_type = None

SPLIT = 'train'

# Set the same audio cache directory as in the training script
AUDIO_CACHE_DIR = "./audio_cache"
os.makedirs(AUDIO_CACHE_DIR, exist_ok=True)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")
logger = logging.getLogger(__name__)


def find_available_models():
    """Find all available model files"""
    model_files = []
    search_paths = ["../models", "."]
    for path in search_paths:
        if os.path.exists(path):
            for file in os.listdir(path):
                if file.endswith('.pt'):
                    full_path = os.path.join(path, file)
                    if os.path.isfile(full_path):
                        model_files.append(full_path)
    return sorted(list(set(model_files)))


def detect_model_type(model_file: str):
    """Detect model family with priority: dccav3 > dccav2 > dcca/dccae > finetuned"""
    filename = os.path.basename(model_file).lower()
    if 'dccav3' in filename:
        return 'dccav3'
    if 'dccav2' in filename:
        return 'dccav2'
    if 'dccae' in filename or 'dcca' in filename:
        return 'dcca'
    return 'finetuned'


def l2_normalize(x: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + eps)


def test_dcca_model(model, text_model, device):
    """Variance test adapted from human_eval to ensure outputs are non-degenerate."""
    logger.info("Testing DCCA-family model functionality...")
    test_descriptions = [
        "A deep male voice",
        "A high female voice",
        "A child's voice",
        "An elderly person speaking"
    ]
    text_outputs = []
    for desc in test_descriptions:
        raw_text_embedding = text_model.encode([desc], convert_to_tensor=True, device=device)
        with torch.no_grad():
            text_output = model.encode_text(raw_text_embedding.to(device)).cpu().numpy()
            text_outputs.append(text_output)
            logger.info(f"Text '{desc}' -> mean {text_output.mean():.4f}, std {text_output.std():.4f}")
    text_outputs = np.array(text_outputs)
    if text_outputs.std() < 1e-6:
        logger.warning("Text encoder outputs nearly identical.")
        return False
    dummy_audio_embeddings = [np.random.randn(1, 256).astype(np.float32) for _ in range(3)]
    audio_outputs = []
    for i, emb in enumerate(dummy_audio_embeddings):
        with torch.no_grad():
            out = model.encode_speech(torch.tensor(emb, dtype=torch.float32, device=device)).cpu().numpy()
            audio_outputs.append(out)
            logger.info(f"Audio {i} -> mean {out.mean():.4f}, std {out.std():.4f}")
    audio_outputs = np.array(audio_outputs)
    if audio_outputs.std() < 1e-6:
        logger.warning("Audio encoder outputs nearly identical.")
        return False
    logger.info("DCCA-family model variance test passed.")
    return True


@st.cache_resource
def load_cotts_dataset():
    """
    Load the CoTTS dataset the same way as in the training script
    """
    try:
        logger.info("Loading CoTTS dataset from disk")
        dataset = load_from_disk(ENRICHED_DATASET_V2_PATH)
        
        # Check if it's already a DatasetDict or a regular Dataset
        if hasattr(dataset, 'keys') and 'train' in dataset.keys():
            # It's already a DatasetDict
            split_dataset = dataset
            # Cast audio column for both splits
            split_dataset['train'] = split_dataset['train'].cast_column("audio", Audio())
            split_dataset['test'] = split_dataset['test'].cast_column("audio", Audio())
        else:
            # It's a regular Dataset, need to split
            dataset = dataset.cast_column("audio", Audio())
            
            # Remove segment_id if it exists, keep other columns as-is
            columns_to_remove = []
            if ID_COLUMN in dataset.column_names:
                columns_to_remove.append(ID_COLUMN)
            
            if columns_to_remove:
                dataset = dataset.remove_columns(columns_to_remove)
            
            # Rename description column if needed
            if "description" in dataset.column_names and DESCRIPTION_COLUMN not in dataset.column_names:
                dataset = dataset.rename_column("description", DESCRIPTION_COLUMN)
            
            split_dataset = dataset.train_test_split(test_size=0.1, seed=42)

        logger.info(
            f"✅ Successfully loaded dataset with {len(split_dataset['train'])} train and {len(split_dataset['test'])} test samples")
        return split_dataset
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        st.error(f"Failed to load dataset: {e}")
        return None


def load_audio_file(audio_data):
    """
    Loads audio from various sources
    """
    try:
        # Handle HuggingFace dataset audio format
        if isinstance(audio_data, dict) and "array" in audio_data and "sampling_rate" in audio_data:
            wav = np.array(audio_data["array"])
            if wav.ndim > 1:  # Multi-channel audio
                wav = np.mean(wav, axis=0)
            return wav, audio_data.get("sampling_rate", 16000)

        # Handle file path
        elif isinstance(audio_data, dict) and "path" in audio_data:
            audio_path = audio_data["path"]
            if os.path.exists(audio_path):
                waveform, sample_rate = torchaudio.load(audio_path)
                wav = waveform.mean(dim=0).numpy()
                return wav, sample_rate

        # Try generic loading
        elif audio_data is not None:
            if hasattr(audio_data, "array"):
                wav = np.array(audio_data.array)
                if wav.ndim > 1:
                    wav = np.mean(wav, axis=0)
                return wav, getattr(audio_data, "sampling_rate", 16000)

    except Exception as e:
        st.error(f"Error loading audio: {e}")

    return np.zeros(16000), 16000  # Return empty audio as fallback


@st.cache_resource
def load_model(model_file):
    """Load DCCA-family (v1/v2/v3) or fine-tuned model."""
    if not model_file or not os.path.exists(model_file):
        return None, None, None, None
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model_type = detect_model_type(model_file)

    def extract_state_dict(obj):
        if isinstance(obj, dict):
            for k in ["model_state_dict", "state_dict", "model"]:
                if k in obj and isinstance(obj[k], dict):
                    return obj[k]
            if all(isinstance(v, torch.Tensor) for v in obj.values()):
                return obj
        return None

    try:
        checkpoint = torch.load(model_file, map_location=device)
    except Exception as e:
        st.error(f"Failed loading checkpoint: {e}")
        return None, None, None, None

    if model_type in {"dcca", "dccav2", "dccav3"}:
        state_dict = extract_state_dict(checkpoint)
        try:
            if model_type == 'dccav3':
                if not HAS_DCCAV3:
                    st.error("dccav3 detected but module missing.")
                    return None, None, None, None
                model = create_dcca_v3_model(state_dict=state_dict)
            elif model_type == 'dccav2':
                if not HAS_DCCAV2:
                    st.error("dccav2 detected but module missing.")
                    return None, None, None, None
                model = create_dcca_v2_model(state_dict=state_dict)
            else:
                model = create_dcca_model(state_dict=state_dict)
            model.to(device).eval()
            desc_embedder = DCCA_DESCRIPTION_EMBEDDER
            test_dcca_model(model, desc_embedder, device)
            return model, desc_embedder, device, model_type
        except Exception as e:
            st.error(f"Failed constructing DCCA-family model: {e}")
            return None, None, None, None

    # Fine-tuned fallback
    model = Voice2Embedding()
    state_dict = extract_state_dict(checkpoint)
    if state_dict:
        try:
            model.load_state_dict(state_dict)
        except Exception as e:
            st.warning(f"Could not load state dict: {e}")
    model.to(device).eval()
    return model, VOICE2EMBEDDING_DESCRIPTION_EMBEDDER, device, 'finetuned'


def extract_audio_embedding(audio_data, model, device, model_type="finetuned"):
    try:
        if model_type in {"dcca", "dccav2", "dccav3"}:
            logger.warning("Audio embedding extraction for DCCA-family should use precomputed embeddings from dataset.")
            return None
        wav, sample_rate = load_audio_file(audio_data)
        if wav is None or len(wav) == 0:
            return None
        mel_spec = torch.tensor(wav_to_mel_spectrogram(wav)).unsqueeze(0).to(device)
        with torch.no_grad():
            return model(mel_spec).cpu().numpy()
    except Exception as e:
        st.error(f"Audio processing error: {e}")
        return None


def extract_text_embedding(text, text_model, model_type="finetuned"):
    with torch.no_grad():
        emb = text_model.encode([text], convert_to_tensor=True)
        return emb.cpu().numpy()


def cosine_similarity(emb1, emb2):
    if emb1 is None or emb2 is None:
        return 0
    t1 = torch.tensor(emb1, dtype=torch.float32)
    t2 = torch.tensor(emb2, dtype=torch.float32)
    if t1.shape[1] != t2.shape[1]:
        return 0
    return F.cosine_similarity(t1, t2, dim=1).mean().item()


def compare_voice_to_text(model, text_model, device, model_type, sample_dataset):
    selected_sample = sample_dataset[st.session_state.selected_idx]
    text_input = st.session_state.text_input
    if model_type in {"dcca", "dccav2", "dccav3"}:
        if RESEMBLYZER_SPEAKER_EMBEDDING_COLUMN not in selected_sample:
            st.error("Missing resemblyzer speaker embedding in sample.")
            st.session_state.similarity_result = None
            return
        raw_audio_emb = np.array(selected_sample[RESEMBLYZER_SPEAKER_EMBEDDING_COLUMN], dtype=np.float32).reshape(1, -1)
        raw_audio_t = torch.from_numpy(raw_audio_emb).to(device)
        with torch.no_grad():
            audio_proj = l2_normalize(model.encode_speech(raw_audio_t)).cpu().numpy()
            raw_text_emb = text_model.encode([text_input], convert_to_tensor=True, device=device)
            text_proj = l2_normalize(model.encode_text(raw_text_emb.to(device))).cpu().numpy()
        similarity = cosine_similarity(audio_proj, text_proj)
    else:
        audio_emb = extract_audio_embedding(selected_sample[AUDIO_COLUMN], model, device, model_type)
        text_emb = extract_text_embedding(text_input, text_model, model_type)
        similarity = cosine_similarity(audio_emb, text_emb) if (audio_emb is not None and text_emb is not None) else None
    st.session_state.similarity_result = similarity


st.title("Voice-to-Text Embedding Demo")
st.write("Compare voice characteristics with text descriptions using DCCA-family (v1/v2/v3) or fine-tuned models.")

# Model selection UI
available_models = find_available_models()
if not available_models:
    st.error("No model files (.pt) found.")
    st.stop()

model_options = {}
for mf in available_models:
    mtype = detect_model_type(mf)
    display = f"{os.path.basename(mf)} ({mtype})"
    model_options[display] = mf

selected_model_display = st.selectbox("Choose a model", list(model_options.keys()))
selected_model_file = model_options[selected_model_display]
if st.session_state.selected_model_file != selected_model_file:
    st.session_state.selected_model_file = selected_model_file
    st.session_state.current_model_type = detect_model_type(selected_model_file)
    st.rerun()

model, text_model, device, model_type = load_model(st.session_state.selected_model_file)
if model is None:
    st.error("Failed to load model.")
    st.stop()

st.info(f"Using model: {os.path.basename(st.session_state.selected_model_file)}  | type: {model_type}")

# Load dataset (was missing causing 'dataset' undefined)
dataset = load_cotts_dataset()
if dataset is None:
    st.error("Dataset failed to load.")
    st.stop()

# Get a sample of the dataset
sample_size = min(100, len(dataset['train']))

# Use consistent sampling with fixed seed for stability
random.seed(213)
sample_indices = random.sample(range(len(dataset['train'])), sample_size)
sample_dataset = dataset['train'].select(sample_indices)

# Create display options with indices
audio_options = {f"Sample {i + 1} - {sample[DESCRIPTION_COLUMN][:50]}...": i
                 for i, sample in enumerate(sample_dataset)}

def update_selected_idx_callback():
    """Update the selected index without triggering a refresh"""
    selected_option = st.session_state.sample_selector
    st.session_state.selected_idx = audio_options[selected_option]

    # Update the text input with the new sample's description
    selected_sample = sample_dataset[st.session_state.selected_idx]
    st.session_state.text_input = selected_sample[DESCRIPTION_COLUMN]

# Use on_change callback to update session state without form submission
st.selectbox(
    "Choose an audio sample",
    list(audio_options.keys()),
    key="sample_selector",
    on_change=update_selected_idx_callback,
    index=list(audio_options.keys()).index(next((k for k, v in audio_options.items()
                                                 if v == st.session_state.selected_idx),
                                                list(audio_options.keys())[0]))
)

# Get the selected sample using the stored index
selected_sample = sample_dataset[st.session_state.selected_idx]

# Display sample information
st.subheader("Current sample")
st.write(f"**Speaker description:** {selected_sample[DESCRIPTION_COLUMN]}")

# Process and display audio
wav, sample_rate = load_audio_file(selected_sample[AUDIO_COLUMN])
if len(wav) > 0:
    # For streamed data, create a temporary file
    temp_path = os.path.join(AUDIO_CACHE_DIR, f"temp_audio_{st.session_state.selected_idx}.wav")
    sf.write(temp_path, wav, sample_rate)
    st.audio(temp_path)
else:
    st.error("Unable to play audio for this sample")

# Create a form to prevent auto-submission on Enter key
with st.form(key="text_comparison_form"):
    # Get user input with the current sample's description as default
    if not st.session_state.text_input and selected_sample[DESCRIPTION_COLUMN]:
        st.session_state.text_input = selected_sample[DESCRIPTION_COLUMN]

    st.text_area("Enter a description of the voice",
                 key="text_input",
                 height=100)

    # Form submission button
    submit_button = st.form_submit_button("Compare Voice to Text")
    
    if submit_button:
        compare_voice_to_text(model, text_model, device, model_type, sample_dataset)

# Display results (outside the form)
if st.session_state.similarity_result is not None:
    similarity = st.session_state.similarity_result

    # Display results
    st.success(f"Processing complete!")
    st.metric("Cosine Similarity", f"{similarity:.4f}")

    # Interpretation
    if similarity > 0.7:
        st.write("✅ High similarity - the description matches the voice well!")
    elif similarity > 0.4:
        st.write("🟡 Moderate similarity - the description partially matches the voice.")
    else:
        st.write("❌ Low similarity - the description doesn't match the voice well.")
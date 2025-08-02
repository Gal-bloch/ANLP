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
from Voice2Embedding import Voice2Embedding
from create_dataset import ENRICHED_DATASET_PATH, AUDIO_COLUMN, DESCRIPTION_COLUMN, GRANITE_DESCRIPTION_EMBEDDING_COLUMN

# Initialize session state
if 'selected_idx' not in st.session_state:
    st.session_state.selected_idx = 0
if 'text_input' not in st.session_state:
    st.session_state.text_input = ""
if 'similarity_result' not in st.session_state:
    st.session_state.similarity_result = None

SPLIT = 'train'

# Set the same audio cache directory as in the training script
AUDIO_CACHE_DIR = "./audio_cache"
os.makedirs(AUDIO_CACHE_DIR, exist_ok=True)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")
logger = logging.getLogger(__name__)


@st.cache_resource
def load_cotts_dataset():
    """
    Load the CoTTS dataset the same way as in the training script
    """
    try:
        logger.info("Loading CoTTS dataset from disk")
        dataset = load_from_disk(ENRICHED_DATASET_PATH)
        dataset = dataset.cast_column(AUDIO_COLUMN, Audio())

        # Split into train and validation (same as in training)
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
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    dense_embedding_model = SentenceTransformer("ibm-granite/granite-embedding-125m-english")
    voice_encoder = VoiceEncoder(device=device)
    model = Voice2Embedding(voice_encoder, projection_dim=dense_embedding_model.get_sentence_embedding_dimension())
    try:
        checkpoint = torch.load("../models/best_voice2embedding_model.pt", map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info("Successfully loaded model weights")
    except Exception as e:
        logger.warning(f"Could not load model weights: {e}. Using uninitialized model.")
    model.to(device)
    model.eval()
    return model, dense_embedding_model, device


def extract_audio_embedding(audio_data, model, device):
    try:
        wav, sample_rate = load_audio_file(audio_data)
        if wav is None or len(wav) == 0:
            st.error("Failed to load audio data")
            return None

        mel_spec = torch.tensor(wav_to_mel_spectrogram(wav)).unsqueeze(0).to(device)
        with torch.no_grad():
            audio_embedding = model(mel_spec).cpu().numpy()
        return audio_embedding
    except Exception as e:
        st.error(f"Error processing audio: {e}")
        return None

def cosine_similarity(emb1, emb2):
    if emb1 is None or emb2 is None or emb1.shape[1] != emb2.shape[1]:
        return 0  # Return zero similarity for invalid inputs

    # Convert to PyTorch tensors
    tensor1 = torch.tensor(emb1, dtype=torch.float32)
    tensor2 = torch.tensor(emb2, dtype=torch.float32)

    # Compute cosine similarity along the feature dimension
    sim = F.cosine_similarity(tensor1, tensor2, dim=1)

    # Return similarity score (not distance)
    return sim.mean().item()


def update_selected_idx():
    """Update the selected index without triggering a refresh"""
    selected_option = st.session_state.sample_selector
    st.session_state.selected_idx = audio_options[selected_option]

    # Update the text input with the new sample's description
    selected_sample = sample_dataset[st.session_state.selected_idx]
    st.session_state.text_input = selected_sample[DESCRIPTION_COLUMN]


def update_text_input():
    """Keep track of text input changes"""
    # This just stores the current text input value in session state
    pass  # The actual update happens automatically through the key


def compare_voice_to_text():
    """Process the comparison when the button is clicked"""
    selected_sample = sample_dataset[st.session_state.selected_idx]
    text_input = st.session_state.text_input

    # Extract embeddings
    audio_embedding = extract_audio_embedding(selected_sample[AUDIO_COLUMN], model, device)
    text_embedding = selected_sample[GRANITE_DESCRIPTION_EMBEDDING_COLUMN]

    if audio_embedding is not None and text_embedding is not None:
        # Calculate similarity
        similarity = cosine_similarity(audio_embedding, text_embedding)

        # Store in session state
        st.session_state.similarity_result = similarity
    else:
        st.session_state.similarity_result = None


st.title("Voice-to-Text Embedding Demo")
st.write("Select an instance from the CoTTS dataset to compare voice characteristics with text descriptions.")

# Load the model and dataset
model, text_model, device = load_model()
dataset = load_cotts_dataset()

if dataset is None:
    st.error("Failed to load dataset. Please check your dataset path.")
else:
    # Get a sample of the dataset
    sample_size = min(100, len(dataset['train']))

    # Use consistent sampling with fixed seed for stability
    random.seed(42)
    sample_indices = random.sample(range(len(dataset['train'])), sample_size)
    sample_dataset = dataset['train'].select(sample_indices)

    # Create display options with indices
    audio_options = {f"Sample {i + 1} - {sample[DESCRIPTION_COLUMN][:50]}...": i
                     for i, sample in enumerate(sample_dataset)}

    # Use on_change callback to update session state without form submission
    st.selectbox(
        "Choose an audio sample",
        list(audio_options.keys()),
        key="sample_selector",
        on_change=update_selected_idx,
        index=list(audio_options.keys()).index(next((k for k, v in audio_options.items()
                                                     if v == st.session_state.selected_idx),
                                                    list(audio_options.keys())[0]))
    )

    # Get the selected sample using the stored index
    selected_sample = sample_dataset[st.session_state.selected_idx]

    # Display sample information
    st.subheader("Current sample")
    st.write(f"**Speaker description:** {selected_sample['text_description']}")

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
        submit_button = st.form_submit_button("Compare Voice to Text", on_click=compare_voice_to_text)

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
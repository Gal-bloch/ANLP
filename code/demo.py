import streamlit as st
import torch
import torchaudio
import numpy as np
import random
from sentence_transformers import SentenceTransformer
from resemblyzer import VoiceEncoder, wav_to_mel_spectrogram
import torch.nn.functional as F
from itertools import islice
import soundfile as sf
import os
import logging
from datasets import load_dataset, Dataset, concatenate_datasets
from voice_to_embedding import Voice2Embedding

SPLIT = 'train'

# Set the same audio cache directory as in the training script
AUDIO_CACHE_DIR = "./audio_cache"
os.makedirs(AUDIO_CACHE_DIR, exist_ok=True)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")
logger = logging.getLogger(__name__)


@st.cache_resource
def merge_datasets_with_audio(id_column="original_path"):
    """
    Replicating the exact same function from train.py
    """
    dataset_name = "parler-tts/mls_eng"
    metadata_dataset_name = "parler-tts/mls-eng-speaker-descriptions"

    logger.info(f"Loading audio dataset: {dataset_name}")
    # Load as streaming but convert to a dictionary of limited datasets
    loader = load_dataset(dataset_name, streaming=True)

    # Create a dict to hold limited data by split
    dataset = {}
    for split_name, max_examples in zip([SPLIT], [100]):
        logger.info(f"Taking first {max_examples} examples from '{split_name}' split")
        split_data = list(islice(loader[split_name], max_examples))
        # Convert the list of examples to a dataset
        dataset[split_name] = Dataset.from_list(split_data)

    logger.info(f"Loading metadata dataset: {metadata_dataset_name}")
    metadata_dataset = load_dataset(metadata_dataset_name)

    merged_data = {}

    for split in dataset:
        if split in metadata_dataset:
            logger.info(f"Merging '{split}' split...")

            # Limit metadata to match the audio dataset size
            metadata_split = metadata_dataset[split].select(
                range(min(len(dataset[split]), len(metadata_dataset[split]))))

            # Get the list of columns from both datasets
            audio_columns = dataset[split].column_names
            metadata_columns = metadata_split.column_names

            # Find common columns that need to be renamed in metadata dataset
            duplicated_columns = [col for col in metadata_columns if col in audio_columns and col != id_column]

            # Rename the duplicated columns in metadata dataset
            for col in duplicated_columns:
                metadata_split = metadata_split.rename_column(col, f"metadata_{col}")

            # Rename the ID column to make validation easier
            metadata_split = metadata_split.rename_column(id_column, f"metadata_{id_column}")

            # Concatenate datasets
            merged_split = concatenate_datasets([dataset[split], metadata_split], axis=1)

            # Validate alignment
            if len(merged_split.filter(lambda id1, id2: id1 != id2,
                                       input_columns=[id_column, f"metadata_{id_column}"])) != 0:
                raise ValueError(f"Mismatch in IDs after merging on split {split}")

            merged_data[split] = merged_split
        else:
            logger.warning(f"Split '{split}' is missing from the metadata dataset.")

    logger.info("✅ Dataset successfully merged!")
    return merged_data


def load_audio_file(audio_data):
    """
    Loads audio from various sources, matching the approach in the training script.
    Handles both file paths and streamed audio data.
    """
    try:
        # If it's a path, try to load it
        if isinstance(audio_data, dict) and "path" in audio_data:
            audio_path = audio_data["path"]
            if os.path.exists(audio_path):
                waveform, sample_rate = torchaudio.load(audio_path)
                wav = waveform.mean(dim=0).numpy()
                return wav, audio_path

        # If it's streamed data or array data
        if isinstance(audio_data, dict) and "array" in audio_data and "sampling_rate" in audio_data:
            wav = np.array(audio_data["array"])
            if wav.ndim > 1:  # Multi-channel audio
                wav = np.mean(wav, axis=0)
            return wav, None

        # Try using the raw audio data if available
        if hasattr(audio_data, "array") and hasattr(audio_data, "sampling_rate"):
            wav = np.array(audio_data["array"])
            if wav.ndim > 1:  # Multi-channel audio
                wav = np.mean(wav, axis=0)
            return wav, None

    except Exception as e:
        st.error(f"Error loading audio: {e}")

    return np.zeros(16000), None  # Return empty audio as fallback


@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dense_embedding_model = SentenceTransformer("ibm-granite/granite-embedding-125m-english")
    voice_encoder = VoiceEncoder(device=device)
    model = Voice2Embedding(voice_encoder, projection_dim=dense_embedding_model.get_sentence_embedding_dimension())
    try:
        checkpoint = torch.load("best_voice2embedding_model.pt", map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
    except Exception as e:
        st.error(f"Error loading model: {e}")
    model.to(device)
    model.eval()
    return model, dense_embedding_model, device


@st.cache_resource
def load_test_samples():
    try:
        # Use the exact same loading method as train.py
        dataset = merge_datasets_with_audio()
        if SPLIT not in dataset:
            st.error("Test split not found in dataset")
            return []

        # Take a sample of the dev set
        test_samples = dataset[SPLIT].select(range(min(100, len(dataset[SPLIT]))))
        samples = random.sample(list(test_samples), min(100, len(test_samples)))

        # Debug information
        st.write(f"Successfully loaded {len(samples)} samples")
        return samples
    except Exception as e:
        st.error(f"Error loading samples: {e}")
        return []


def extract_audio_embedding(audio_data, model, device):
    try:
        wav, _ = load_audio_file(audio_data)
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


def extract_text_embedding(text, text_model):
    with torch.no_grad():
        text_embedding = text_model.encode([text], convert_to_tensor=True).cpu().numpy()
    return text_embedding


def cosine_similarity(emb1, emb2):
    if emb1 is None or emb2 is None or emb1.shape[1] != emb2.shape[1]:
        return np.array([[0]])  # Return zero similarity for invalid inputs

    # Convert to PyTorch tensors
    tensor1 = torch.tensor(emb1, dtype=torch.float32)
    tensor2 = torch.tensor(emb2, dtype=torch.float32)

    # Compute cosine similarity along the feature dimension
    sim = F.cosine_similarity(tensor1, tensor2, dim=1)

    # Return 1 - mean similarity, wrapped as a numpy array
    return np.array([[1 - sim.mean().item()]])


st.title("Voice-to-Text Embedding Demo")
st.write("Select an instance from the dataset or upload your own audio file.")

model, text_model, device = load_model()
samples = load_test_samples()

if not samples:
    st.error("No samples loaded. Please check your dataset configuration.")
else:
    audio_options = {f"Sample {i + 1} - {sample.get('text_description', 'No description')}": sample
                     for i, sample in enumerate(samples)}
    selected_sample = st.selectbox("Choose an audio sample", list(audio_options.keys()))
    selected_audio = audio_options[selected_sample]

    # Display the audio if possible
    wav, audio_path = load_audio_file(selected_audio.get("audio", {}))
    if audio_path and os.path.exists(audio_path):
        st.audio(audio_path)
    elif len(wav) > 0:
        # For streamed data, create a temporary file
        temp_path = os.path.join(AUDIO_CACHE_DIR, "temp_audio.wav")
        sf.write(temp_path, wav, 16000)
        st.audio(temp_path)
    else:
        st.error("Unable to play audio for this sample")

    st.write(f"Text description: {selected_audio.get('text_description', 'No description')}")

    text_input = st.text_input("Enter a description of the voice")

    if text_input:
        st.write("Processing the audio...")
        audio_embedding = extract_audio_embedding(selected_audio.get("audio", {}), model, device)
        if audio_embedding is not None:
            text_embedding = extract_text_embedding(text_input, text_model)
            similarity = cosine_similarity(audio_embedding, text_embedding)
            st.write(f"Cosine Similarity between Voice and Text: {similarity[0][0]:.4f}")
        else:
            st.error("Error processing audio file.")
    else:
        st.write("Please enter a text description.")
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
from voice_to_embedding import Voice2Embedding
from tempfile import NamedTemporaryFile

# Initialize session state
if 'search_text' not in st.session_state:
    st.session_state.search_text = ""
if 'uploaded_audio' not in st.session_state:
    st.session_state.uploaded_audio = None
if 'search_results' not in st.session_state:
    st.session_state.search_results = None
if 'search_type' not in st.session_state:
    st.session_state.search_type = "text"
if 'num_results' not in st.session_state:
    st.session_state.num_results = 5
if 'precomputed_embeddings' not in st.session_state:
    st.session_state.precomputed_embeddings = None

# Set the audio cache directory
AUDIO_CACHE_DIR = "./audio_cache"
os.makedirs(AUDIO_CACHE_DIR, exist_ok=True)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")
logger = logging.getLogger(__name__)


@st.cache_resource
def load_cotts_dataset():
    """
    Load the CoTTS dataset
    """
    try:
        logger.info("Loading CoTTS dataset from disk")
        dataset = load_from_disk("/Users/galbloch/Desktop/school/git/ANLP/datasets/CoTTS_dataset")
        dataset = dataset.cast_column("audio", Audio())

        # Rename + clean the same way as in training
        dataset = dataset.rename_column("description", "text_description").remove_columns(["segment_id"])

        # Split into train and validation
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

        # Handle uploaded file
        elif isinstance(audio_data, bytes):
            with NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
                tmp.write(audio_data)
                tmp.flush()
                waveform, sample_rate = torchaudio.load(tmp.name)
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dense_embedding_model = SentenceTransformer("ibm-granite/granite-embedding-125m-english")
    voice_encoder = VoiceEncoder(device=device)
    model = Voice2Embedding(voice_encoder, projection_dim=dense_embedding_model.get_sentence_embedding_dimension())
    try:
        checkpoint = torch.load("best_voice2embedding_model.pt", map_location=device)
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


def extract_text_embedding(text, text_model):
    with torch.no_grad():
        text_embedding = text_model.encode([text], convert_to_tensor=True).cpu().numpy()
    return text_embedding


def cosine_similarity(emb1, emb2):
    if emb1 is None or emb2 is None or emb1.shape[1] != emb2.shape[1]:
        return 0  # Return zero similarity for invalid inputs

    # Convert to PyTorch tensors
    tensor1 = torch.tensor(emb1, dtype=torch.float32)
    tensor2 = torch.tensor(emb2, dtype=torch.float32)

    # Compute cosine similarity along the feature dimension
    sim = F.cosine_similarity(tensor1, tensor2, dim=1)

    # Return similarity score
    return sim.mean().item()


# Note the underscore prefix to prevent Streamlit from trying to hash the dataset
def precompute_voice_embeddings(_dataset, model, device):
    """
    Precompute voice embeddings for all samples in the dataset
    Returns the precomputed embeddings
    """
    logger.info("Precomputing voice embeddings for all samples...")

    # Limit dataset size for demo/development
    limited_dataset = _dataset['test'].select(range(10000))

    # Store precomputed embeddings
    precomputed_embeddings = []
    total_samples = len(limited_dataset)

    # Create progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, sample in enumerate(limited_dataset):
        # Update progress bar
        if i % 10 == 0:
            progress = min(i / total_samples, 1.0)
            progress_bar.progress(progress)
            status_text.text(f"Processing sample {i}/{total_samples} ({progress * 100:.1f}%)")

        # Extract audio embedding
        audio_embedding = extract_audio_embedding(sample['audio'], model, device)

        # Store embedding with sample info
        if audio_embedding is not None:
            precomputed_embeddings.append({
                'index': i,
                'description': sample['text_description'],
                'audio': sample['audio'],
                'embedding': audio_embedding
            })

    # Complete progress bar
    progress_bar.progress(1.0)
    status_text.text(f"✅ Precomputed {len(precomputed_embeddings)} voice embeddings")
    logger.info(f"✅ Successfully precomputed {len(precomputed_embeddings)} voice embeddings")

    return precomputed_embeddings


def find_similar_voices():
    """
    Find voices similar to the query (text or audio) using precomputed embeddings
    """
    # Reset results when new search is performed
    st.session_state.search_results = None

    search_type = st.session_state.search_type
    num_results = st.session_state.num_results
    precomputed_embeddings = st.session_state.precomputed_embeddings

    # Get query embedding based on search type
    query_embedding = None

    if search_type == "text" and st.session_state.search_text:
        query_embedding = extract_text_embedding(st.session_state.search_text, text_model)
    elif search_type == "audio" and st.session_state.uploaded_audio:
        query_embedding = extract_audio_embedding(st.session_state.uploaded_audio, model, device)

    if query_embedding is None:
        st.error("Please provide a valid text query or upload an audio file")
        return

    if precomputed_embeddings is None or len(precomputed_embeddings) == 0:
        st.error("No precomputed embeddings available. Please try reloading the application.")
        return

    # Use precomputed embeddings to find similar voices
    results = []
    for item in precomputed_embeddings:
        similarity = cosine_similarity(query_embedding, item['embedding'])
        results.append({
            'index': item['index'],
            'description': item['description'],
            'similarity': similarity,
            'audio': item['audio']
        })

    # Sort results by similarity (highest first)
    results.sort(key=lambda x: x['similarity'], reverse=True)

    # Take the top N results
    st.session_state.search_results = results[:num_results]


st.title("Voice Search Engine")
st.write("Find similar voices using text descriptions or audio uploads")

# Load model and dataset
model, text_model, device = load_model()
dataset = load_cotts_dataset()

if dataset is None:
    st.error("Failed to load dataset. Please check your dataset path.")
else:
    # Precompute voice embeddings for the dataset if not already done
    if st.session_state.precomputed_embeddings is None:
        with st.spinner("Precomputing voice embeddings... This may take a few minutes."):
            st.session_state.precomputed_embeddings = precompute_voice_embeddings(dataset, model, device)
        st.success(f"Successfully precomputed {len(st.session_state.precomputed_embeddings)} voice embeddings!")

    # Create tabs for different search methods
    search_tab, about_tab = st.tabs(["Search", "About"])

    with search_tab:
        # Search method selection
        st.radio(
            "Search method",
            options=["Find voices by text description", "Find voices by audio upload"],
            key="search_method",
            horizontal=True,
            on_change=lambda: setattr(st.session_state, "search_type",
                                      "text" if st.session_state.search_method == "Find voices by text description" else "audio")
        )

        # Number of results to show
        st.slider("Number of results", min_value=1, max_value=20, key="num_results")

        # Create search form
        with st.form(key="voice_search_form"):
            if st.session_state.search_type == "text":
                st.text_area(
                    "Describe the voice you're looking for",
                    placeholder="Example: A deep male voice with a slight accent and slow speaking pace",
                    key="search_text",
                    height=100
                )
                uploaded_file = None
            else:
                uploaded_file = st.file_uploader("Upload an audio sample of the voice (WAV, MP3, or OGG format)",
                                                 type=["wav", "mp3", "ogg"])
                if uploaded_file:
                    st.audio(uploaded_file)
                    # Read file contents
                    st.session_state.uploaded_audio = uploaded_file.read()

            # Form submission button
            submit_button = st.form_submit_button("Search for Similar Voices", on_click=find_similar_voices)

        # Display results
        if st.session_state.search_results:
            st.subheader("Search Results")

            for i, result in enumerate(st.session_state.search_results):
                with st.expander(f"Result #{i + 1}: Similarity {result['similarity']:.4f}"):
                    st.write(f"**Description:** {result['description']}")

                    # Process and display audio
                    wav, sample_rate = load_audio_file(result['audio'])
                    if len(wav) > 0:
                        # For streamed data, create a temporary file
                        temp_path = os.path.join(AUDIO_CACHE_DIR, f"result_audio_{i}.wav")
                        sf.write(temp_path, wav, sample_rate)
                        st.audio(temp_path)
                    else:
                        st.error("Unable to play audio for this sample")

                    # Show similarity interpretation
                    if result['similarity'] > 0.7:
                        st.write("✅ Very high match")
                    elif result['similarity'] > 0.5:
                        st.write("🟢 Good match")
                    elif result['similarity'] > 0.3:
                        st.write("🟡 Moderate match")
                    else:
                        st.write("🔴 Low match")

    with about_tab:
        st.header("About this Demo")
        st.write("""
        This demo showcases a voice search engine built with a voice-to-embedding model.

        ### How it works
        1. The model converts both voices and text descriptions into the same embedding space
        2. You can search using either text descriptions or by uploading your own audio
        3. The system finds voices that are most similar to your query

        ### Use cases
        - Voice casting for films, games, or animation
        - Finding voice actors with specific characteristics
        - Exploring different voice types for text-to-speech applications
        - Research on voice perception and description

        ### Technical details
        The model uses a Voice Encoder and a Text Embedding model, connected by a projection layer
        that aligns voice characteristics with their textual descriptions.
        """)
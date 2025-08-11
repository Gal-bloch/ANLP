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
try:
    # Optional import for V3 models
    from DCCAV3 import create_dcca_v3_model
    HAS_DCCAV3 = True
except ImportError:
    HAS_DCCAV3 = False
from create_dataset import (
    ENRICHED_DATASET_V2_PATH, 
    AUDIO_COLUMN, 
    DESCRIPTION_COLUMN, 
    GRANITE_DESCRIPTION_EMBEDDING_COLUMN,
    RESEMBLYZER_SPEAKER_EMBEDDING_COLUMN,
    ID_COLUMN
)
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
if 'selected_model_file' not in st.session_state:
    st.session_state.selected_model_file = None
if 'current_model_type' not in st.session_state:
    st.session_state.current_model_type = None

# Set the audio cache directory
AUDIO_CACHE_DIR = "./audio_cache"
os.makedirs(AUDIO_CACHE_DIR, exist_ok=True)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")
logger = logging.getLogger(__name__)


def find_available_models():
    """Find all available model files"""
    model_files = []
    
    # Search in the models directory and current directory
    search_paths = [
        "../models",
        "."
    ]
    
    for path in search_paths:
        if os.path.exists(path):
            for file in os.listdir(path):
                if file.endswith('.pt'):
                    full_path = os.path.join(path, file)
                    if os.path.isfile(full_path):
                        model_files.append(full_path)
    
    # Remove duplicates and sort
    model_files = sorted(list(set(model_files)))
    return model_files


def detect_model_type(model_file):
    """Detect if model is DCCA or fine-tuned based on filename"""
    filename = os.path.basename(model_file).lower()
    if 'dccav3' in filename:
        return "dccav3"
    if 'dccae' in filename or 'dcca' in filename:
        return "dcca"
    else:
        return "finetuned"


def test_dcca_model(model, text_model, device, model_type="dcca"):
    """Test if DCCA model is working properly by checking output variance"""
    logger.info(f"Testing {model_type.upper()} model functionality...")
    
    # Create some test text inputs
    test_descriptions = [
        "A deep male voice",
        "A high female voice", 
        "A child's voice",
        "An elderly person speaking"
    ]
    
    # Get text embeddings and pass through model
    text_outputs = []
    for desc in test_descriptions:
        raw_text_embedding = text_model.encode([desc], convert_to_tensor=True, device=device)
        with torch.no_grad():
            # Process like in classifier evaluation - squeeze and ensure proper device
            text_tensor = raw_text_embedding.to(device)
            text_output = model.encode_text(text_tensor).cpu().numpy()
            text_outputs.append(text_output)
            logger.info(f"Text '{desc}' -> embedding mean: {text_output.mean():.4f}, std: {text_output.std():.4f}")
    
    # Check if outputs are different
    text_outputs = np.array(text_outputs)
    output_std = text_outputs.std()
    logger.info(f"Overall text output std: {output_std:.4f}")
    
    # Model is working if outputs vary (std > small threshold)
    return output_std > 1e-6


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
def load_model(model_file):
    """Load either DCCA or fine-tuned model based on model file"""
    if not model_file or not os.path.exists(model_file):
        return None, None, None, None
        
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    
    model_type = detect_model_type(model_file)
    
    if model_type in ["dcca", "dccav3"]:
        # Load DCCA-family model
        logger.info(f"Loading {model_type.upper()} model from {model_file}...")
        try:
            checkpoint = torch.load(model_file, map_location=device)
            logger.info(f"Checkpoint keys: {list(checkpoint.keys()) if isinstance(checkpoint, dict) else 'Not a dict'}")
            
            # Helper to extract a state dict
            def extract_state_dict(obj):
                if isinstance(obj, dict):
                    for k in ["model_state_dict", "state_dict", "model"]:
                        if k in obj and isinstance(obj[k], dict):
                            return obj[k]
                    if all(isinstance(v, torch.Tensor) for v in obj.values()):
                        return obj  # looks like raw state dict
                return None

            state_dict = extract_state_dict(checkpoint)

            if model_type == "dccav3":
                if not HAS_DCCAV3:
                    st.error("DCCAV3 model file detected but DCCAV3 module not importable.")
                    return None, None, None, None
                model = create_dcca_v3_model(state_dict=state_dict)
            else: # dcca
                model = create_dcca_model(state_dict=state_dict)

            model.to(device)
            model.eval()

            # Test if the model is working properly
            model_works = test_dcca_model(model, DCCA_DESCRIPTION_EMBEDDER, device, model_type)
            if not model_works:
                logger.warning(f"⚠️ {model_type.upper()} model test failed - results may be unreliable")
            
            logger.info(f"✅ Successfully loaded {model_type.upper()} model")
            return model, DCCA_DESCRIPTION_EMBEDDER, device, model_type
            
        except Exception as e:
            logger.error(f"Failed to load {model_type.upper()} model: {e}")
            st.error(f"Failed to load {model_type.upper()} model: {e}")
            return None, None, None, None
    
    else:  # finetuned model
        # Load fine-tuned Voice2Embedding model
        logger.info(f"Loading fine-tuned Voice2Embedding model from {model_file}...")
        voice_encoder = VoiceEncoder(device=device)
        dense_embedding_model = SentenceTransformer("ibm-granite/granite-embedding-125m-english")
        model = Voice2Embedding(voice_encoder, projection_dim=dense_embedding_model.get_sentence_embedding_dimension())
        
        try:
            checkpoint = torch.load(model_file, map_location=device)
            logger.info(f"Fine-tuned checkpoint keys: {list(checkpoint.keys()) if isinstance(checkpoint, dict) else 'Not a dict'}")
            
            # Try different ways to extract the model state dict
            if isinstance(checkpoint, dict):
                if "model_state_dict" in checkpoint:
                    state_dict = checkpoint["model_state_dict"]
                elif "state_dict" in checkpoint:
                    state_dict = checkpoint["state_dict"]
                elif "model" in checkpoint:
                    state_dict = checkpoint["model"]
                else:
                    # Assume the checkpoint is the state dict itself
                    state_dict = checkpoint
                
                model.load_state_dict(state_dict)
                logger.info("✅ Successfully loaded fine-tuned model weights")
            else:
                logger.warning("Checkpoint is not a dictionary, using uninitialized model.")
                
        except Exception as e:
            logger.warning(f"Could not load model weights: {e}. Using uninitialized model.")
        
        model.to(device)
        model.eval()
        
        return model, VOICE2EMBEDDING_DESCRIPTION_EMBEDDER, device, "finetuned"


def extract_audio_embedding(audio_data, model, device, model_type="finetuned"):
    """Extract audio embedding using either DCCA or fine-tuned model"""
    try:
        if model_type in ["dcca", "dccav3"]:
            # For DCCA-family, we expect precomputed embeddings in the dataset
            # This function is mainly for compatibility
            logger.warning(f"extract_audio_embedding called for {model_type.upper()} model - should use precomputed embeddings")
            return None
        
        # For fine-tuned model, compute embeddings in real-time
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


def extract_text_embedding(text, text_model, model_type="finetuned"):
    """Extract text embedding - works for both model types"""
    with torch.no_grad():
        if model_type in ["dcca", "dccav3"]:
            # For DCCA-family, encode text and then pass through model
            text_embedding = text_model.encode([text], convert_to_tensor=True)
            return text_embedding.cpu().numpy()
        else:
            # For fine-tuned model, just get the text embedding
            text_embedding = text_model.encode([text], convert_to_tensor=True).cpu().numpy()
            return text_embedding


def cosine_similarity(emb1, emb2):
    if emb1 is None or emb2 is None:
        return 0  # Return zero similarity for None inputs
    
    # Convert to PyTorch tensors
    tensor1 = torch.tensor(emb1, dtype=torch.float32)
    tensor2 = torch.tensor(emb2, dtype=torch.float32)
    
    # Ensure tensors are 2D
    if tensor1.dim() == 1:
        tensor1 = tensor1.unsqueeze(0)
    if tensor2.dim() == 1:
        tensor2 = tensor2.unsqueeze(0)
    
    # Check if embeddings have compatible dimensions
    if tensor1.shape[1] != tensor2.shape[1]:
        return 0  # Return zero similarity for incompatible dimensions

    # Compute cosine similarity along the feature dimension
    sim = F.cosine_similarity(tensor1, tensor2, dim=1)

    # Return similarity score
    return sim.mean().item()


# Note the underscore prefix to prevent Streamlit from trying to hash the dataset
def precompute_voice_embeddings(_dataset, model, device, model_type="finetuned"):
    """
    Precompute voice embeddings for all samples in the dataset
    Returns the precomputed embeddings
    """
    logger.info("Precomputing voice embeddings for all samples...")

    # Limit dataset size for demo/development
    test_dataset = _dataset['test']
    max_samples = min(10000, len(test_dataset))
    limited_dataset = test_dataset.select(range(max_samples))

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

        # Extract audio embedding based on model type
        if model_type in ["dcca", "dccav3"]:
            # For DCCA model, use precomputed resemblyzer embeddings from dataset
            if RESEMBLYZER_SPEAKER_EMBEDDING_COLUMN in sample:
                # Use precomputed resemblyzer embeddings from dataset
                audio_embedding = np.array(sample[RESEMBLYZER_SPEAKER_EMBEDDING_COLUMN]).reshape(1, -1)
                
                # Process audio embedding through DCCA model
                audio_embedding_tensor = torch.tensor(audio_embedding, dtype=torch.float32).to(device)
                with torch.no_grad():
                    processed_audio_embedding = model.encode_speech(audio_embedding_tensor).cpu().numpy()
                    # Ensure embedding is 2D
                    if processed_audio_embedding.ndim == 1:
                        processed_audio_embedding = processed_audio_embedding.reshape(1, -1)
            else:
                logger.warning(f"No precomputed audio embeddings for sample {i}")
                continue
        else:
            # For fine-tuned model, extract embeddings normally
            audio_column = AUDIO_COLUMN if AUDIO_COLUMN in sample else 'audio'
            processed_audio_embedding = extract_audio_embedding(sample[audio_column], model, device, model_type)

        # Store embedding with sample info
        if processed_audio_embedding is not None:
            description_column = DESCRIPTION_COLUMN if DESCRIPTION_COLUMN in sample else 'text_description'
            audio_column = AUDIO_COLUMN if AUDIO_COLUMN in sample else 'audio'
            
            precomputed_embeddings.append({
                'index': i,
                'description': sample[description_column],
                'audio': sample[audio_column],
                'embedding': processed_audio_embedding
            })

    # Complete progress bar
    progress_bar.progress(1.0)
    status_text.text(f"✅ Precomputed {len(precomputed_embeddings)} voice embeddings")
    logger.info(f"✅ Successfully precomputed {len(precomputed_embeddings)} voice embeddings")

    return precomputed_embeddings


def find_similar_voices(model, text_model, device):
    """
    Find voices similar to the query (text or audio) using precomputed embeddings
    """
    # Reset results when new search is performed
    st.session_state.search_results = None

    search_type = st.session_state.search_type
    num_results = st.session_state.num_results
    precomputed_embeddings = st.session_state.precomputed_embeddings
    model_type = st.session_state.current_model_type

    # Get query embedding based on search type
    query_embedding = None

    if search_type == "text" and st.session_state.search_text:
        if model_type in ["dcca", "dccav3"]:
            # For DCCA, encode text and pass through model
            try:
                raw_text_embedding = text_model.encode([st.session_state.search_text], convert_to_tensor=True, device=device)
                with torch.no_grad():
                    text_tensor = raw_text_embedding.to(device)
                    query_embedding = model.encode_text(text_tensor).cpu().numpy()
                    # Ensure query embedding is 2D
                    if query_embedding.ndim == 1:
                        query_embedding = query_embedding.reshape(1, -1)
                logger.info(f"{model_type.upper()} text query embedding shape: {query_embedding.shape}")
            except Exception as e:
                st.error(f"Error processing text with {model_type.upper()} model: {e}")
                return
        else:
            # For fine-tuned model, use regular text embedding
            query_embedding = extract_text_embedding(st.session_state.search_text, text_model, model_type)
            logger.info(f"Fine-tuned text query embedding shape: {query_embedding.shape if query_embedding is not None else 'None'}")
            
    elif search_type == "audio" and st.session_state.uploaded_audio:
        if model_type in ["dcca", "dccav3"]:
            # For DCCA, would need to extract resemblyzer embedding from uploaded audio and process it
            # This is more complex and might require additional processing
            st.error(f"Audio search with {model_type.upper()} models is not yet fully implemented for uploaded files")
            return
        else:
            # For fine-tuned model, extract embedding normally
            query_embedding = extract_audio_embedding(st.session_state.uploaded_audio, model, device, model_type)
            logger.info(f"Fine-tuned audio query embedding shape: {query_embedding.shape if query_embedding is not None else 'None'}")

    if query_embedding is None:
        st.error("Please provide a valid text query or upload an audio file")
        return

    if precomputed_embeddings is None or len(precomputed_embeddings) == 0:
        st.error("No precomputed embeddings available. Please try reloading the application.")
        return

    # Use precomputed embeddings to find similar voices
    results = []
    for item in precomputed_embeddings:
        try:
            similarity = cosine_similarity(query_embedding, item['embedding'])
            results.append({
                'index': item['index'],
                'description': item['description'],
                'similarity': similarity,
                'audio': item['audio']
            })
        except Exception as e:
            logger.warning(f"Error calculating similarity for item {item['index']}: {e}")
            # Skip this item and continue
            continue

    # Sort results by similarity (highest first)
    results.sort(key=lambda x: x['similarity'], reverse=True)

    # Take the top N results
    st.session_state.search_results = results[:num_results]


st.title("Voice Search Engine")
st.write("Find similar voices using text descriptions or audio uploads")

# Model selection
st.subheader("Model Selection")
available_models = find_available_models()

if not available_models:
    st.error("No model files (.pt) found. Please ensure you have model files in the '../models' directory or current directory.")
    st.stop()

# Display available models with their types
model_options = {}
for model_file in available_models:
    model_name = os.path.basename(model_file)
    model_type = detect_model_type(model_file)
    display_name = f"{model_name} ({model_type.upper()})"
    model_options[display_name] = model_file

selected_model_display = st.selectbox("Choose a model", list(model_options.keys()))
selected_model_file = model_options[selected_model_display]

# Load model if changed
if st.session_state.selected_model_file != selected_model_file:
    st.session_state.selected_model_file = selected_model_file
    st.session_state.current_model_type = detect_model_type(selected_model_file)
    # Clear precomputed embeddings when model changes
    st.session_state.precomputed_embeddings = None
    # Clear cached model by forcing a rerun
    st.rerun()

# Load model and dataset
model, text_model, device, model_type = load_model(st.session_state.selected_model_file)
st.session_state.current_model_type = model_type
dataset = load_cotts_dataset()

if model is None:
    st.error("Failed to load the selected model. Please check the model file.")
    st.stop()

if dataset is None:
    st.error("Failed to load dataset. Please check your dataset path.")
    st.stop()

# Display current model info
st.info(f"Using {model_type.upper()} model: {os.path.basename(st.session_state.selected_model_file)}")

# Precompute voice embeddings for the dataset if not already done
if st.session_state.precomputed_embeddings is None:
    with st.spinner("Precomputing voice embeddings... This may take a few minutes."):
        st.session_state.precomputed_embeddings = precompute_voice_embeddings(dataset, model, device, model_type)
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
            # Show warning for DCCA models with audio upload
            if model_type in ["dcca", "dccav3"]:
                st.warning("⚠️ Audio upload search is limited with DCCA models. Text search is recommended.")
            
            uploaded_file = st.file_uploader("Upload an audio sample of the voice (WAV, MP3, or OGG format)",
                                             type=["wav", "mp3", "ogg"])
            if uploaded_file:
                st.audio(uploaded_file)
                # Read file contents
                st.session_state.uploaded_audio = uploaded_file.read()

        # Form submission button
        submit_button = st.form_submit_button("Search for Similar Voices")
        
        if submit_button:
            find_similar_voices(model, text_model, device)

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
    st.write(f"""
    This demo showcases a voice search engine built with {'DCCA-family' if model_type in ['dcca', 'dccav3'] else 'fine-tuned'} voice-to-embedding models.

    ### Current Model: {model_type.upper()}
    {'**DCCA (Deep Canonical Correlation Analysis)**: Maps voice and text to a shared embedding space using correlation maximization.' if model_type in ['dcca', 'dccav3'] else '**Fine-tuned Voice2Embedding**: Direct projection from voice features to text embedding space.'}

    ### How it works
    1. The model converts both voices and text descriptions into the same embedding space
    2. You can search using text descriptions{' (audio upload has limited support for DCCA models)' if model_type in ['dcca', 'dccav3'] else ' or by uploading your own audio'}
    3. The system finds voices that are most similar to your query

    ### Use cases
    - Voice casting for films, games, or animation
    - Finding voice actors with specific characteristics
    - Exploring different voice types for text-to-speech applications
    - Research on voice perception and description

    ### Technical details
    The model uses a Voice Encoder and a Text Embedding model, connected by {'DCCA layers that learn optimal correlation between modalities' if model_type in ['dcca', 'dccav3'] else 'a projection layer that aligns voice characteristics with their textual descriptions'}.
    """)
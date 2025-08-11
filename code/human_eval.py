import streamlit as st
import torch
import torchaudio
import numpy as np
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
    RESEMBLYZER_SPEAKER_EMBEDDING_COLUMN, 
    GRANITE_DESCRIPTION_EMBEDDING_COLUMN,
    ENRICHED_DATASET_V2_PATH,
    DESCRIPTION_COLUMN,
    ID_COLUMN,
    AUDIO_COLUMN
)
from tempfile import NamedTemporaryFile
import json
import pickle

# Initialize session state for human evaluation
if 'precomputed_embeddings' not in st.session_state:
    st.session_state.precomputed_embeddings = None
if 'selected_model_file' not in st.session_state:
    st.session_state.selected_model_file = None
if 'character_descriptions' not in st.session_state:
    st.session_state.character_descriptions = [
        "A deep male voice with a slight accent and slow speaking pace",
        "A high-pitched female voice with cheerful tone",
        "An elderly man with a raspy, weathered voice",
        "A young woman with a soft, gentle speaking style",
        "A middle-aged male with authoritative, clear pronunciation",
        "A child's voice, innocent and playful",
        "A dramatic female voice with theatrical delivery",
        "A calm, soothing male narrator voice",
        "An energetic young female with fast speech",
        "A wise elderly woman with measured speech"
    ]
if 'current_description_idx' not in st.session_state:
    st.session_state.current_description_idx = 0
if 'current_results' not in st.session_state:
    st.session_state.current_results = []
if 'current_result_idx' not in st.session_state:
    st.session_state.current_result_idx = 0
if 'evaluation_data' not in st.session_state:
    st.session_state.evaluation_data = []
if 'total_evaluations' not in st.session_state:
    st.session_state.total_evaluations = 0
if 'correct_evaluations' not in st.session_state:
    st.session_state.correct_evaluations = 0
if 'evaluation_complete' not in st.session_state:
    st.session_state.evaluation_complete = False

# Set the audio cache directory and embeddings cache file
AUDIO_CACHE_DIR = "./audio_cache"
def get_embeddings_cache_file(model_file):
    """Generate cache filename based on model file"""
    model_name = os.path.splitext(os.path.basename(model_file))[0]
    return f"./embeddings_cache_{model_name}.pkl"


def find_available_models():
    """Find all available model files"""
    model_files = []
    
    # Search only in the specific models directory
    search_paths = [
        "../models"
    ]
    
    for path in search_paths:
        if os.path.exists(path):
            for file in os.listdir(path):
                if file.endswith('.pt'):  # Simply check for .pt extension
                    full_path = os.path.join(path, file)
                    if os.path.isfile(full_path):
                        model_files.append(full_path)
    
    # Remove duplicates and sort
    model_files = sorted(list(set(model_files)))
    return model_files


def detect_model_type(model_file):
    """Detect model family based on filename.
    Priority order: dccav3 > dccae/dcca > finetuned
    """
    filename = os.path.basename(model_file).lower()
    if 'dccav3' in filename:
        return 'dccav3'
    if 'dccae' in filename or 'dcca' in filename:
        return 'dcca'
    return 'finetuned'

os.makedirs(AUDIO_CACHE_DIR, exist_ok=True)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")
logger = logging.getLogger(__name__)


def save_embeddings_to_cache(embeddings, cache_file):
    """Save precomputed embeddings to cache file"""
    try:
        logger.info(f"Saving {len(embeddings)} embeddings to cache file: {cache_file}")
        with open(cache_file, 'wb') as f:
            pickle.dump(embeddings, f)
        logger.info("✅ Successfully saved embeddings to cache")
        return True
    except Exception as e:
        logger.error(f"Failed to save embeddings to cache: {e}")
        return False


def load_embeddings_from_cache(cache_file):
    """Load precomputed embeddings from cache file"""
    try:
        if not os.path.exists(cache_file):
            logger.info(f"Cache file not found: {cache_file}")
            return None

        logger.info(f"Loading embeddings from cache file: {cache_file}")
        with open(cache_file, 'rb') as f:
            embeddings = pickle.load(f)
        logger.info(f"✅ Successfully loaded {len(embeddings)} embeddings from cache")
        return embeddings
    except Exception as e:
        logger.error(f"Failed to load embeddings from cache: {e}")
        return None


def get_cache_info(cache_file):
    """Get information about the cache file"""
    if not os.path.exists(cache_file):
        return None

    try:
        stat = os.stat(cache_file)
        size_mb = stat.st_size / (1024 * 1024)
        modified_time = stat.st_mtime

        # Try to get the number of embeddings without loading the full file
        with open(cache_file, 'rb') as f:
            embeddings = pickle.load(f)
            count = len(embeddings) if embeddings else 0

        return {
            'size_mb': size_mb,
            'modified_time': modified_time,
            'count': count
        }
    except Exception as e:
        logger.error(f"Error getting cache info: {e}")
        return None


@st.cache_resource
def load_cotts_dataset():
    """Load the CoTTS dataset"""
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
    """Loads audio from various sources"""
    try:
        if isinstance(audio_data, dict) and "array" in audio_data and "sampling_rate" in audio_data:
            wav = np.array(audio_data["array"])
            if wav.ndim > 1:
                wav = np.mean(wav, axis=0)
            return wav, audio_data.get("sampling_rate", 16000)
        elif isinstance(audio_data, dict) and "path" in audio_data:
            audio_path = audio_data["path"]
            if os.path.exists(audio_path):
                waveform, sample_rate = torchaudio.load(audio_path)
                wav = waveform.mean(dim=0).numpy()
                return wav, sample_rate
        elif isinstance(audio_data, bytes):
            with NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
                tmp.write(audio_data)
                tmp.flush()
                waveform, sample_rate = torchaudio.load(tmp.name)
                wav = waveform.mean(dim=0).numpy()
                return wav, sample_rate
        elif audio_data is not None:
            if hasattr(audio_data, "array"):
                wav = np.array(audio_data.array)
                if wav.ndim > 1:
                    wav = np.mean(wav, axis=0)
                return wav, getattr(audio_data, "sampling_rate", 16000)
    except Exception as e:
        st.error(f"Error loading audio: {e}")
    return np.zeros(16000), 16000


@st.cache_resource
def load_model(model_file):
    """Load a model: supports DCCA (v1/v2), DCCAV3, or fine-tuned Voice2Embedding."""
    if not model_file or not os.path.exists(model_file):
        return None, None, None, None

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    model_type = detect_model_type(model_file)
    logger.info(f"Detected model type: {model_type}")

    try:
        checkpoint = torch.load(model_file, map_location=device)
        logger.info(f"Checkpoint keys: {list(checkpoint.keys()) if isinstance(checkpoint, dict) else 'Not a dict'}")
    except Exception as e:
        st.error(f"Failed to load checkpoint: {e}")
        return None, None, None, None

    # Helper to extract a state dict
    def extract_state_dict(obj):
        if isinstance(obj, dict):
            for k in ["model_state_dict", "state_dict", "model"]:
                if k in obj and isinstance(obj[k], dict):
                    return obj[k]
            if all(isinstance(v, torch.Tensor) for v in obj.values()):
                return obj  # looks like raw state dict
        return None

    if model_type in {"dcca", "dccav3"}:
        state_dict = extract_state_dict(checkpoint)
        try:
            if model_type == "dccav3":
                if not HAS_DCCAV3:
                    st.error("DCCAV3 model file detected but DCCAV3 module not importable.")
                    return None, None, None, None
                model = create_dcca_v3_model(state_dict=state_dict)
            else:  # original dcca
                model = create_dcca_model(state_dict=state_dict)

            model.to(device)
            model.eval()

            # Use shared description embedder (same Granite model)
            desc_embedder = DCCA_DESCRIPTION_EMBEDDER

            # Functional test
            if not test_dcca_model(model, desc_embedder, device):
                logger.warning("⚠️ Model functional variance test failed (embeddings near-constant).")

            logger.info("✅ Loaded DCCA-family model successfully")
            return model, desc_embedder, device, model_type
        except Exception as e:
            st.error(f"Failed to construct DCCA-family model: {e}")
            logger.exception(e)
            return None, None, None, None

    # Fine-tuned Voice2Embedding fallback
    logger.info(f"Loading fine-tuned Voice2Embedding model from {model_file}...")
    model = Voice2Embedding()
    state_dict = extract_state_dict(checkpoint)
    if state_dict is not None:
        try:
            model.load_state_dict(state_dict)
            logger.info("✅ Loaded fine-tuned model weights")
        except Exception as e:
            logger.warning(f"Could not load model weights: {e}; using base initialization.")
    else:
        logger.warning("Checkpoint did not contain a recognizable state dict; using base initialization.")
    model.to(device)
    model.eval()
    return model, VOICE2EMBEDDING_DESCRIPTION_EMBEDDER, device, "finetuned"


def extract_audio_embedding(audio_data, model, device, model_type="finetuned"):
    """Extract audio embedding using either DCCA or fine-tuned model"""
    try:
        if model_type in {"dcca", "dccav3"}:
            # For DCCA-family, we expect precomputed embeddings in the dataset
            logger.warning("extract_audio_embedding called for DCCA-family model - should use precomputed embeddings")
            return None

        # Fine-tuned: compute embeddings in real-time
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
        if model_type in {"dcca", "dccav3"}:
            text_embedding = text_model.encode([text], convert_to_tensor=True)
            return text_embedding.cpu().numpy()
        # Fine-tuned model
        text_embedding = text_model.encode([text], convert_to_tensor=True).cpu().numpy()
        return text_embedding


def cosine_similarity(emb1, emb2):
    if emb1 is None or emb2 is None:
        return 0
    tensor1 = torch.tensor(emb1, dtype=torch.float32)
    tensor2 = torch.tensor(emb2, dtype=torch.float32)
    if tensor1.shape[1] != tensor2.shape[1]:
        return 0

    sim = F.cosine_similarity(tensor1, tensor2, dim=1)
    return sim.mean().item()


def precompute_voice_embeddings(_dataset, model, device, model_type="finetuned", model_file=None):
    """Precompute voice embeddings for all samples in the dataset"""
    logger.info(f"Preparing voice embeddings for {model_type} model...")
    
    # Use the full test dataset, but limit to available samples
    test_dataset = _dataset['test']
    max_samples = min(len(test_dataset), 1000)  # Use available samples, max 1000
    limited_dataset = test_dataset.select(range(max_samples))
    
    precomputed_embeddings = []
    total_samples = len(limited_dataset)

    progress_bar = st.progress(0)
    status_text = st.empty()

    if model_type in {"dcca", "dccav3"}:
        # For DCCA, use precomputed embeddings from the dataset
        logger.info("Using precomputed embeddings from dataset for DCCA model")
        
        for i, sample in enumerate(limited_dataset):
            if i % 10 == 0:
                progress = min(i / total_samples, 1.0)
                progress_bar.progress(progress)
                status_text.text(f"Processing sample {i}/{total_samples} ({progress * 100:.1f}%)")

            # Use precomputed resemblyzer embeddings from dataset
            if RESEMBLYZER_SPEAKER_EMBEDDING_COLUMN in sample:
                audio_embedding = np.array(sample[RESEMBLYZER_SPEAKER_EMBEDDING_COLUMN]).reshape(1, -1)
                
                precomputed_embeddings.append({
                    'index': i,
                    'description': sample[DESCRIPTION_COLUMN],
                    'audio': sample[AUDIO_COLUMN],
                    'embedding': audio_embedding,
                    'precomputed_text_embedding': np.array(sample.get(GRANITE_DESCRIPTION_EMBEDDING_COLUMN, [])).reshape(1, -1) if GRANITE_DESCRIPTION_EMBEDDING_COLUMN in sample else None
                })
    else:
        # For fine-tuned model, compute embeddings in real-time
        logger.info("Computing embeddings in real-time for fine-tuned model")
        
        for i, sample in enumerate(limited_dataset):
            if i % 10 == 0:
                progress = min(i / total_samples, 1.0)
                progress_bar.progress(progress)
                status_text.text(f"Processing sample {i}/{total_samples} ({progress * 100:.1f}%)")

            audio_embedding = extract_audio_embedding(sample['audio'], model, device, model_type)
            if audio_embedding is not None:
                precomputed_embeddings.append({
                    'index': i,
                    'description': sample[DESCRIPTION_COLUMN],
                    'audio': sample[AUDIO_COLUMN],
                    'embedding': audio_embedding
                })

    progress_bar.progress(1.0)
    status_text.text(f"✅ Prepared {len(precomputed_embeddings)} voice embeddings")
    logger.info(f"✅ Successfully prepared {len(precomputed_embeddings)} voice embeddings for {model_type} model")

    # Save to cache
    if model_file:
        cache_file = get_embeddings_cache_file(model_file)
        save_embeddings_to_cache(precomputed_embeddings, cache_file)

    return precomputed_embeddings


def search_for_description(description, top_k=3, model_type="finetuned", model=None, text_model=None, device=None):
    """Search for voices matching a description"""
    
    if model_type in {"dcca", "dccav3"}:
        # Encode text then pass through model encoder; preserve batch dim
        raw_text_embedding = text_model.encode([description], convert_to_tensor=True, device=device)
        with torch.no_grad():
            text_tensor = raw_text_embedding.to(device)  # shape (1, dim)
            query_embedding = model.encode_text(text_tensor).cpu().numpy()
        logger.info(f"DCCA-family query embedding shape: {query_embedding.shape}, mean: {query_embedding.mean():.4f}, std: {query_embedding.std():.4f}")
    else:
        # For fine-tuned model, just get text embedding
        query_embedding = extract_text_embedding(description, text_model, model_type)
    
    if query_embedding is None:
        return []

    results = []
    similarities = []  # Debug: collect similarities
    for i, item in enumerate(st.session_state.precomputed_embeddings):
        if model_type in {"dcca", "dccav3"}:
            if 'embedding' in item:
                audio_embedding_tensor = torch.tensor(item['embedding'], dtype=torch.float32).to(device)  # (1, 256)
                with torch.no_grad():
                    processed_audio_embedding = model.encode_speech(audio_embedding_tensor).cpu().numpy()
                if i < 3:
                    logger.info(f"DCCA-family audio emb {i} -> processed shape {processed_audio_embedding.shape}, mean {processed_audio_embedding.mean():.4f}, std {processed_audio_embedding.std():.4f}")
                similarity = cosine_similarity(query_embedding, processed_audio_embedding)
                similarities.append(similarity)
            else:
                similarity = 0
        else:
            # For fine-tuned model, use embeddings directly
            similarity = cosine_similarity(query_embedding, item['embedding'])
            similarities.append(similarity)
            
        results.append({
            'index': item['index'],
            'description': item['description'],
            'similarity': similarity,
            'audio': item['audio']
        })

    # Debug: Print similarity statistics
    if similarities:
        similarities_array = np.array(similarities)
        logger.info(f"Similarity stats - min: {similarities_array.min():.4f}, max: {similarities_array.max():.4f}, mean: {similarities_array.mean():.4f}, std: {similarities_array.std():.4f}")
        logger.info(f"Unique similarities: {len(np.unique(similarities_array))}/{len(similarities_array)}")

    results.sort(key=lambda x: x['similarity'], reverse=True)
    return results[:top_k]


def get_next_evaluation(model_type="finetuned", model=None, text_model=None, device=None):
    """Get the next audio sample to evaluate"""
    if st.session_state.current_description_idx >= len(st.session_state.character_descriptions):
        st.session_state.evaluation_complete = True
        return None, None, None

    # If we need new results for current description
    if not st.session_state.current_results or st.session_state.current_result_idx >= len(
            st.session_state.current_results):
        if st.session_state.current_result_idx >= len(st.session_state.current_results):
            # Move to next description
            st.session_state.current_description_idx += 1
            st.session_state.current_result_idx = 0

            if st.session_state.current_description_idx >= len(st.session_state.character_descriptions):
                st.session_state.evaluation_complete = True
                return None, None, None

        # Get results for current description
        current_desc = st.session_state.character_descriptions[st.session_state.current_description_idx]
        st.session_state.current_results = search_for_description(
            current_desc, top_k=3, model_type=model_type, model=model, 
            text_model=text_model, device=device
        )

    if st.session_state.current_result_idx < len(st.session_state.current_results):
        current_desc = st.session_state.character_descriptions[st.session_state.current_description_idx]
        current_result = st.session_state.current_results[st.session_state.current_result_idx]
        return current_desc, current_result, st.session_state.current_result_idx

    return None, None, None


def record_evaluation(matches):
    """Record user evaluation"""
    st.session_state.total_evaluations += 1
    if matches:
        st.session_state.correct_evaluations += 1

    # Store evaluation data
    current_desc = st.session_state.character_descriptions[st.session_state.current_description_idx]
    current_result = st.session_state.current_results[st.session_state.current_result_idx]

    st.session_state.evaluation_data.append({
        'description': current_desc,
        'audio_description': current_result['description'],
        'similarity_score': current_result['similarity'],
        'user_approval': matches,
        'result_rank': st.session_state.current_result_idx + 1
    })

    # Move to next result
    st.session_state.current_result_idx += 1


def save_evaluation_results():
    """Save evaluation results to file"""
    results = {
        'total_evaluations': st.session_state.total_evaluations,
        'correct_evaluations': st.session_state.correct_evaluations,
        'accuracy': st.session_state.correct_evaluations / st.session_state.total_evaluations if st.session_state.total_evaluations > 0 else 0,
        'detailed_results': st.session_state.evaluation_data
    }

    with open('human_evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    return results


# Add after the other utility functions

def test_dcca_model(model, text_model, device):
    """Test if DCCA model is working properly by checking output variance"""
    logger.info("Testing DCCA model functionality...")
    
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
        raw_text_embedding = text_model.encode([desc], convert_to_tensor=True, device=device)  # keep batch dim
        with torch.no_grad():
            text_output = model.encode_text(raw_text_embedding.to(device)).cpu().numpy()
            text_outputs.append(text_output)
            logger.info(f"Text '{desc}' -> embedding mean: {text_output.mean():.4f}, std: {text_output.std():.4f}")
    
    # Check if outputs are different
    text_outputs = np.array(text_outputs)
    output_std = text_outputs.std()
    logger.info(f"Overall text output std: {output_std:.4f}")
    
    if output_std < 1e-6:
        logger.warning("⚠️ DCCA text encoder outputs are nearly identical - model may not be trained!")
        return False
    
    # Test with some dummy audio embeddings
    dummy_audio_embeddings = [
        np.random.randn(1, 256).astype(np.float32),
        np.random.randn(1, 256).astype(np.float32),
        np.random.randn(1, 256).astype(np.float32),
    ]
    
    audio_outputs = []
    for i, audio_emb in enumerate(dummy_audio_embeddings):
        # Process like in classifier evaluation - squeeze and ensure proper device
        audio_tensor = torch.tensor(audio_emb, dtype=torch.float32).to(device)  # (1, 256)
        with torch.no_grad():
            audio_output = model.encode_speech(audio_tensor).cpu().numpy()
            audio_outputs.append(audio_output)
            logger.info(f"Audio {i} -> embedding mean: {audio_output.mean():.4f}, std: {audio_output.std():.4f}")
    
    audio_outputs = np.array(audio_outputs)
    audio_output_std = audio_outputs.std()
    logger.info(f"Overall audio output std: {audio_output_std:.4f}")
    
    if audio_output_std < 1e-6:
        logger.warning("⚠️ DCCA audio encoder outputs are nearly identical - model may not be trained!")
        return False
    
    logger.info("✅ DCCA model appears to be working correctly")
    return True


# Main Streamlit app
st.title("Voice Search Engine - Human Evaluation")
st.write("Evaluate how well the voice search engine matches descriptions to voices")

# Model selection
st.sidebar.header("Model Configuration")

# Find available models
available_models = find_available_models()

if not available_models:
    st.error("No model files found! Please ensure model files (.pt) are in the current directory or models/ folder.")
    st.stop()

# Create display names for models
model_display_names = []
for model_file in available_models:
    model_name = os.path.basename(model_file)
    model_type = detect_model_type(model_file)
    model_display_names.append(f"{model_name} ({model_type})")

# Model selection dropdown
selected_display_name = st.sidebar.selectbox(
    "Select Model",
    model_display_names,
    help="Choose a model file. DCCA models use precomputed embeddings, others compute embeddings in real-time."
)

# Get the selected model file
selected_model_index = model_display_names.index(selected_display_name)
selected_model_file = available_models[selected_model_index]
model_type = detect_model_type(selected_model_file)

# Update session state if model file changed
if selected_model_file != st.session_state.selected_model_file:
    st.session_state.selected_model_file = selected_model_file
    st.session_state.precomputed_embeddings = None  # Reset embeddings cache
    
    # Reset all evaluation state when switching models
    st.session_state.current_description_idx = 0
    st.session_state.current_results = []
    st.session_state.current_result_idx = 0
    st.session_state.evaluation_data = []
    st.session_state.total_evaluations = 0
    st.session_state.correct_evaluations = 0
    st.session_state.evaluation_complete = False
    
    st.info("🔄 Model changed - evaluation reset. You'll start fresh with the new model.")
    st.rerun()

# Model info
st.sidebar.info(f"**Selected**: {os.path.basename(selected_model_file)}")
if model_type in {"dcca", "dccav3"}:
    sidebar_label = "DCCA V3" if model_type == "dccav3" else "DCCA"
    st.sidebar.info(f"🔬 **{sidebar_label} Model**: Uses precomputed speaker embeddings; re-encoded on the fly for similarity.")
else:
    st.sidebar.info("🎯 **Fine-tuned Model**: Computes embeddings in real-time. More flexible.")

# Load model and dataset
model, text_model, device, loaded_model_type = load_model(selected_model_file)
dataset = load_cotts_dataset()

if dataset is None or model is None:
    st.error("Failed to load dataset or model. Please check your setup.")
else:
    # Get the cache file for the current model
    cache_file = get_embeddings_cache_file(selected_model_file)
    
    # Display cache information
    cache_info = get_cache_info(cache_file)
    if cache_info:
        st.info(f"📁 Embeddings cache found: {cache_info['count']} embeddings ({cache_info['size_mb']:.1f} MB)")

    # Try to load embeddings from cache first
    if st.session_state.precomputed_embeddings is None:
        # Try loading from cache
        cached_embeddings = load_embeddings_from_cache(cache_file)

        if cached_embeddings is not None:
            st.session_state.precomputed_embeddings = cached_embeddings
            st.success(f"✅ Loaded {len(cached_embeddings)} precomputed embeddings from cache!")
        else:
            # Need to compute embeddings
            if model_type == "dcca":
                st.info("📁 DCCA model detected. Loading embeddings from dataset...")
                button_text = "🔄 Load DCCA Embeddings from Dataset"
            else:
                st.warning("No cached embeddings found. Computing embeddings...")
                button_text = "🔄 Compute Fine-tuned Model Embeddings"

            # Add option to clear cache or recompute
            col1, col2 = st.columns(2)
            with col1:
                if st.button(button_text):
                    with st.spinner("Processing embeddings... This may take a few minutes."):
                        st.session_state.precomputed_embeddings = precompute_voice_embeddings(
                            dataset, model, device, model_type, selected_model_file
                        )
                    st.success(
                        f"Successfully processed and cached {len(st.session_state.precomputed_embeddings)} voice embeddings!")
                    st.rerun()

            with col2:
                if st.button("🗑️ Clear Cache") and cache_info:
                    try:
                        os.remove(cache_file)
                        st.success("Cache cleared successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to clear cache: {e}")

            if st.session_state.precomputed_embeddings is None:
                if model_type == "dcca":
                    st.info(f"Click '{button_text}' to load embeddings from the dataset.")
                else:
                    st.info(f"Click '{button_text}' to start the evaluation.")
                st.stop()

    # Display current accuracy
    accuracy = (
            st.session_state.correct_evaluations / st.session_state.total_evaluations * 100) if st.session_state.total_evaluations > 0 else 0
    st.metric("Current Accuracy", f"{accuracy:.1f}%")

    # Progress indicator
    total_possible = len(st.session_state.character_descriptions) * 3  # 3 results per description
    current_progress = st.session_state.total_evaluations
    st.progress(current_progress / total_possible if total_possible > 0 else 0)
    st.write(f"Progress: {current_progress}/{total_possible} evaluations completed")

    if not st.session_state.evaluation_complete:
        # Get next evaluation
        description, result, rank = get_next_evaluation(model_type, model, text_model, device)

        if description and result:
            st.subheader("Evaluation Task")

            # Show the description (what we're looking for)
            st.write("**Target Description:**")
            st.write(f"*{description}*")

            st.write(f"**Retrieved Voice #{rank + 1} (Similarity: {result['similarity']:.4f})**")
            st.write("Listen to this voice and decide if it matches the description above:")

            # Play the audio
            wav, sample_rate = load_audio_file(result['audio'])
            if len(wav) > 0:
                temp_path = os.path.join(AUDIO_CACHE_DIR, f"eval_audio_{st.session_state.total_evaluations}.wav")
                sf.write(temp_path, wav, sample_rate)
                st.audio(temp_path)
            else:
                st.error("Unable to play audio for this sample")

            # Evaluation buttons
            col1, col2 = st.columns(2)

            with col1:
                if st.button("✅ Matches Description", use_container_width=True):
                    record_evaluation(True)
                    st.rerun()

            with col2:
                if st.button("❌ Does Not Match", use_container_width=True):
                    record_evaluation(False)
                    st.rerun()

            # Show some context
            with st.expander("Show actual audio description (for reference only)"):
                st.write(f"*{result['description']}*")
                st.write(
                    "**Note:** This is the original description from the dataset. Base your evaluation only on what you hear and the target description above.")

        else:
            st.error("Unable to get next evaluation sample")

    else:
        # Evaluation complete
        st.success("🎉 Evaluation Complete!")

        final_accuracy = (
                st.session_state.correct_evaluations / st.session_state.total_evaluations * 100) if st.session_state.total_evaluations > 0 else 0

        st.write(f"**Final Results:**")
        st.write(f"- Total Evaluations: {st.session_state.total_evaluations}")
        st.write(f"- Correct Matches: {st.session_state.correct_evaluations}")
        st.write(f"- Final Accuracy: {final_accuracy:.1f}%")

        if st.button("Save Results"):
            results = save_evaluation_results()
            st.success("Results saved to 'human_evaluation_results.json'")
            st.json(results)

        if st.button("Start New Evaluation"):
            # Reset all session state
            for key in ['current_description_idx', 'current_results', 'current_result_idx',
                        'evaluation_data', 'total_evaluations', 'correct_evaluations', 'evaluation_complete']:
                st.session_state[
                    key] = 0 if 'idx' in key or 'evaluations' in key else [] if 'data' in key or 'results' in key else False
            st.rerun()

    # Show evaluation statistics
    with st.expander("Evaluation Statistics"):
        if st.session_state.evaluation_data:
            st.write("**Per Description Breakdown:**")

            # Group by description
            desc_stats = {}
            for item in st.session_state.evaluation_data:
                desc = item['description']
                if desc not in desc_stats:
                    desc_stats[desc] = {'total': 0, 'correct': 0}
                desc_stats[desc]['total'] += 1
                if item['user_approval']:
                    desc_stats[desc]['correct'] += 1

            for desc, stats in desc_stats.items():
                acc = (stats['correct'] / stats['total'] * 100) if stats['total'] > 0 else 0
                st.write(f"- *{desc}*: {stats['correct']}/{stats['total']} ({acc:.1f}%)")
        else:
            st.write("No evaluation data yet.")
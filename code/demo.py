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
    if 'dccae' in filename or 'dcca' in filename:
        return "dcca"
    else:
        return "finetuned"


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
        raw_text_embedding = text_model.encode([desc], convert_to_tensor=True, device=device)
        with torch.no_grad():
            # Process like in classifier evaluation - squeeze and ensure proper device
            text_tensor = raw_text_embedding.squeeze().to(device)
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
    
    if model_type == "dcca":
        # Load DCCA model
        logger.info(f"Loading DCCA model from {model_file}...")
        try:
            checkpoint = torch.load(model_file, map_location=device)
            logger.info(f"Checkpoint keys: {list(checkpoint.keys()) if isinstance(checkpoint, dict) else 'Not a dict'}")
            
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
            else:
                # If checkpoint is not a dict, assume it's the model directly
                state_dict = None
            
            if state_dict is not None:
                model = create_dcca_model(state_dict=state_dict)
            else:
                # Try loading as direct model
                model = checkpoint
                if not hasattr(model, 'eval'):
                    raise ValueError("Loaded object is not a valid model")
            
            model.to(device)
            model.eval()

            # Test if the model is working properly
            model_works = test_dcca_model(model, DCCA_DESCRIPTION_EMBEDDER, device)
            if not model_works:
                logger.warning("⚠️ DCCA model test failed - results may be unreliable")
            
            logger.info("✅ Successfully loaded DCCA model")
            return model, DCCA_DESCRIPTION_EMBEDDER, device, "dcca"
            
        except Exception as e:
            logger.error(f"Failed to load DCCA model: {e}")
            st.error(f"Failed to load DCCA model: {e}")
            return None, None, None, None
    
    else:  # finetuned model
        # Load fine-tuned Voice2Embedding model
        logger.info(f"Loading fine-tuned Voice2Embedding model from {model_file}...")
        model = Voice2Embedding()
        
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
        if model_type == "dcca":
            # For DCCA, we expect precomputed embeddings in the dataset
            # This function is mainly for compatibility
            logger.warning("extract_audio_embedding called for DCCA model - should use precomputed embeddings")
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
        if model_type == "dcca":
            # For DCCA, encode text and then pass through model
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
    # Check if embeddings are valid
    if tensor1.shape[1] != tensor2.shape[1]:
        return 0  # Return zero similarity for invalid inputs

    # Compute cosine similarity along the feature dimension
    sim = F.cosine_similarity(tensor1, tensor2, dim=1)

    # Return similarity score (not distance)
    return sim.mean().item()


def compare_voice_to_text(model, text_model, device, model_type, sample_dataset):
    """Process the comparison when the button is clicked"""
    selected_sample = sample_dataset[st.session_state.selected_idx]
    text_input = st.session_state.text_input

    if model_type == "dcca":
        # For DCCA model, use precomputed embeddings and encode text through DCCA
        if RESEMBLYZER_SPEAKER_EMBEDDING_COLUMN in selected_sample:
            # Use precomputed resemblyzer embeddings from dataset
            audio_embedding = np.array(selected_sample[RESEMBLYZER_SPEAKER_EMBEDDING_COLUMN]).reshape(1, -1)
            
            # Process audio embedding through DCCA model
            audio_embedding_tensor = torch.tensor(audio_embedding, dtype=torch.float32).squeeze().to(device)
            with torch.no_grad():
                processed_audio_embedding = model.encode_speech(audio_embedding_tensor).cpu().numpy()
            
            # Process text through DCCA model
            raw_text_embedding = text_model.encode([text_input], convert_to_tensor=True, device=device)
            with torch.no_grad():
                text_tensor = raw_text_embedding.squeeze().to(device)
                processed_text_embedding = model.encode_text(text_tensor).cpu().numpy()
            
            # Calculate similarity in the shared DCCA space
            similarity = cosine_similarity(
                processed_audio_embedding.reshape(1, -1), 
                processed_text_embedding.reshape(1, -1)
            )
        else:
            st.error("Precomputed audio embeddings not available for DCCA model")
            st.session_state.similarity_result = None
            return
    else:
        # For fine-tuned model, extract embeddings normally
        audio_embedding = extract_audio_embedding(selected_sample[AUDIO_COLUMN], model, device, model_type)
        text_embedding = extract_text_embedding(text_input, text_model, model_type)

        if audio_embedding is not None and text_embedding is not None:
            # Calculate similarity
            similarity = cosine_similarity(audio_embedding, text_embedding)
        else:
            st.session_state.similarity_result = None
            return

    # Store in session state
    st.session_state.similarity_result = similarity


st.title("Voice-to-Text Embedding Demo")
st.write("Compare voice characteristics with text descriptions using either DCCA or fine-tuned models.")

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
    # Clear cached model by forcing a rerun
    st.rerun()

# Load the model and dataset
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
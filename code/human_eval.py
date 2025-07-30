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
import json
import pickle
from pathlib import Path

# Initialize session state for human evaluation
if 'precomputed_embeddings' not in st.session_state:
    st.session_state.precomputed_embeddings = None
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
EMBEDDINGS_CACHE_FILE = "./embeddings_cache.pkl"
os.makedirs(AUDIO_CACHE_DIR, exist_ok=True)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")
logger = logging.getLogger(__name__)


def save_embeddings_to_cache(embeddings, cache_file=EMBEDDINGS_CACHE_FILE):
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


def load_embeddings_from_cache(cache_file=EMBEDDINGS_CACHE_FILE):
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


def get_cache_info(cache_file=EMBEDDINGS_CACHE_FILE):
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
        dataset = load_from_disk("/Users/galbloch/Desktop/school/git/ANLP/datasets/CoTTS_dataset")
        dataset = dataset.cast_column("audio", Audio())
        dataset = dataset.rename_column("description", "text_description").remove_columns(["segment_id"])
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
def load_model():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
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
        return 0
    tensor1 = torch.tensor(emb1, dtype=torch.float32)
    tensor2 = torch.tensor(emb2, dtype=torch.float32)
    sim = F.cosine_similarity(tensor1, tensor2, dim=1)
    return sim.mean().item()


def precompute_voice_embeddings(_dataset, model, device):
    """Precompute voice embeddings for all samples in the dataset"""
    logger.info("Precomputing voice embeddings for all samples...")
    limited_dataset = _dataset['test'].select(range(67000))
    precomputed_embeddings = []
    total_samples = len(limited_dataset)

    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, sample in enumerate(limited_dataset):
        if i % 10 == 0:
            progress = min(i / total_samples, 1.0)
            progress_bar.progress(progress)
            status_text.text(f"Processing sample {i}/{total_samples} ({progress * 100:.1f}%)")

        audio_embedding = extract_audio_embedding(sample['audio'], model, device)
        if audio_embedding is not None:
            precomputed_embeddings.append({
                'index': i,
                'description': sample['text_description'],
                'audio': sample['audio'],
                'embedding': audio_embedding
            })

    progress_bar.progress(1.0)
    status_text.text(f"✅ Precomputed {len(precomputed_embeddings)} voice embeddings")
    logger.info(f"✅ Successfully precomputed {len(precomputed_embeddings)} voice embeddings")

    # Save to cache
    save_embeddings_to_cache(precomputed_embeddings)

    return precomputed_embeddings


def search_for_description(description, top_k=3):
    """Search for voices matching a description"""
    query_embedding = extract_text_embedding(description, text_model)
    if query_embedding is None:
        return []

    results = []
    for item in st.session_state.precomputed_embeddings:
        similarity = cosine_similarity(query_embedding, item['embedding'])
        results.append({
            'index': item['index'],
            'description': item['description'],
            'similarity': similarity,
            'audio': item['audio']
        })

    results.sort(key=lambda x: x['similarity'], reverse=True)
    return results[:top_k]


def get_next_evaluation():
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
        st.session_state.current_results = search_for_description(current_desc, top_k=3)

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


# Main Streamlit app
st.title("Voice Search Engine - Human Evaluation")
st.write("Evaluate how well the voice search engine matches descriptions to voices")

# Load model and dataset
model, text_model, device = load_model()
dataset = load_cotts_dataset()

if dataset is None:
    st.error("Failed to load dataset. Please check your dataset path.")
else:
    # Display cache information
    cache_info = get_cache_info()
    if cache_info:
        st.info(f"📁 Embeddings cache found: {cache_info['count']} embeddings ({cache_info['size_mb']:.1f} MB)")

    # Try to load embeddings from cache first
    if st.session_state.precomputed_embeddings is None:
        # Try loading from cache
        cached_embeddings = load_embeddings_from_cache()

        if cached_embeddings is not None:
            st.session_state.precomputed_embeddings = cached_embeddings
            st.success(f"✅ Loaded {len(cached_embeddings)} precomputed embeddings from cache!")
        else:
            # Need to compute embeddings
            st.warning("No cached embeddings found. Computing embeddings...")

            # Add option to clear cache or recompute
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔄 Recompute Embeddings"):
                    with st.spinner("Computing voice embeddings... This may take a few minutes."):
                        st.session_state.precomputed_embeddings = precompute_voice_embeddings(dataset, model, device)
                    st.success(
                        f"Successfully computed and cached {len(st.session_state.precomputed_embeddings)} voice embeddings!")
                    st.rerun()

            with col2:
                if st.button("🗑️ Clear Cache") and cache_info:
                    try:
                        os.remove(EMBEDDINGS_CACHE_FILE)
                        st.success("Cache cleared successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to clear cache: {e}")

            if st.session_state.precomputed_embeddings is None:
                st.info("Click 'Recompute Embeddings' to start the evaluation.")
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
        description, result, rank = get_next_evaluation()

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
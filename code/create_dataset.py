import os
# Ensure Python uses UTF-8 as default if possible
os.environ["PYTHONUTF8"] = "1"

# === Updated Monkey-patch builtins.open for read mode (non-binary) ===
import builtins

# Save the original open
original_open = builtins.open

import numpy as np
import torch
import librosa
from resemblyzer import VoiceEncoder, wav_to_mel_spectrogram
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F
import logging

def patched_open(file, mode='r', buffering=-1, encoding=None, errors=None, newline=None, closefd=True, opener=None):
    # Only force UTF-8 for text read modes if encoding is not already specified.
    # Do not modify binary mode (i.e. if 'b' is in mode).
    if 'r' in mode and encoding is None and 'b' not in mode:
        encoding = 'utf-8'
    return original_open(file, mode, buffering, encoding, errors, newline, closefd, opener)


# Replace the built-in open with our patched version
builtins.open = patched_open

# --------------------------------------------------
import pandas as pd
from datasets import load_dataset, load_from_disk, DatasetDict
from huggingface_hub import login
import ollama

# Constants for your description CSV file.
GIGASPEECH_SPEECH_DESCRIPTION_DATASET_PATH = "../datasets/EN_description_GIGASPEECH.csv"
GIGASPEECH_SPEECH_LABELS_DATASET_PATH = "../datasets/EN_labels_GIGASPEECH.csv"
ID_COLUMN = "segment_id"
DESCRIPTION_COLUMN = "text_description"
NEGATIVE_DESCRIPTION_COLUMN = "negated_description"
GRANITE_DESCRIPTION_EMBEDDING_COLUMN = "granite_description_embedding"
GRANITE_NEGATIVE_DESCRIPTION_EMBEDDING_COLUMN = "granite_negated_description_embedding"
AUDIO_COLUMN = "audio"
RESEMBLYZER_SPEAKER_EMBEDDING_COLUMN = "resemblyzer_speaker_embedding"
GENDER_COLUMN = "gender"
AGE_COLUMN = "age"
SPEED_COLUMN = "speed"
PITCH_COLUMN = "pitch"
ENERGY_COLUMN = "energy"
EMOTION_COLUMN = "emotion"

COMBINED_DATASET_PATH = r"../datasets/VISTA_dataset"
ENRICHED_DATASET_PATH = r"../datasets/Enriched_VISTA_dataset"
ENRICHED_DATASET_V2_PATH = r"../datasets/Enriched_VISTA_dataset_v2"

VALID_DATASET_SIZES = ["xs", "s", "m", "l", "xl"]

HUGGINGFACE_TOKEN = "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"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# Audio and mel spectrogram hyperparameters from Resemblyzer
sampling_rate = 16000
mel_window_length = 25  # In milliseconds
mel_window_step = 10    # In milliseconds
mel_n_channels = 40

speaker_encoder = VoiceEncoder(device=DEVICE)
description_encoder = SentenceTransformer("ibm-granite/granite-embedding-125m-english")

logger = logging.getLogger(__name__)

REFINEMENT_PROMPT_TEMPLATE = """
You will receive a voice description and a set of voice labels.
Your task is to rewrite the description, removing any details that are not strictly related to the speaker's speech 
attributes and the labels provided, while changing the description as little as possible.

Do NOT add any new details or assumptions, and Don't leave ANY detail in the description that is not in the labels, or that is not about the speaker's speech attributes.
Specifically, remove any detail that pertains to subject or field the speaker is speaking about, and focus ONLY on the speaker's speech attributes.
Keep the style, structure, wording and tone of the original description, but limit the content strictly to the labeled attributes and the speaker's speech characteristics.

Here are the labels:
- Gender: {gender}
- Age: {age}
- Speaking speed: {speed}
- Pitch: {pitch}
- Energy: {energy}
- Emotion: {emotion}

Original description:
"{original_description}"

Output ONLY the new refined description — no numbering, no quotes, no explanations. Just the new description.
"""


def wav_to_mel_spectrogram_batch(wavs, sr=sampling_rate):
    """
    Derives mel spectrograms for a batch of preprocessed audio waveforms.
    Handles variable length inputs by padding to the longest sequence.
    
    Args:
        wavs: List of numpy arrays containing the audio waveforms
        sr: Sampling rate (default: sampling_rate from global config)
    
    Returns:
        mel_specs: Tensor of shape (batch_size, max_len, mel_n_channels)
    """
    # Find the length of the longest sequence
    max_len = max(wav.shape[0] for wav in wavs)
    
    # Pad sequences to max_len
    padded_wavs = []
    for wav in wavs:
        pad_len = max_len - wav.shape[0]
        if pad_len > 0:
            # Pad with zeros (silence)
            padded_wav = np.pad(wav, (0, pad_len), mode='constant', constant_values=0)
        else:
            padded_wav = wav
        padded_wavs.append(padded_wav)
    
    # Compute mel spectrograms for the batch
    mel_specs = []
    for wav in padded_wavs:
        frames = librosa.feature.melspectrogram(
            y=wav,
            sr=sr,
            n_fft=int(sr * mel_window_length / 1000),
            hop_length=int(sr * mel_window_step / 1000),
            n_mels=mel_n_channels
        )
        mel_specs.append(frames.astype(np.float32).T)
    
    # Stack into a single tensor
    mel_specs = torch.tensor(np.stack(mel_specs))
    
    return mel_specs

def extract_audio_embeddings_batch(audio_batch):
    """Process a batch of audio samples to extract speaker embeddings."""
    # Convert batch of audio data to waveforms
    wavs = [np.array(audio["array"]) for audio in audio_batch]
    
    # Convert batch to mel spectrograms
    mel_specs = wav_to_mel_spectrogram_batch(wavs)
    mel_specs = mel_specs.to(DEVICE)
    
    # Process the batch through the encoder
    with torch.no_grad():
        audio_embeddings = speaker_encoder(mel_specs).cpu().numpy()
    
    return audio_embeddings

def extract_audio_embedding(audio):
    """
    Process a single audio sample to extract speaker embedding.
    :param audio: Audio data in a format compatible with Resemblyzer (e.g., numpy array).
    :return: Numpy array of the audio embedding.
    """
    wav = np.array(audio["array"])
    if wav.ndim > 1:
        wav = np.mean(wav, axis=0)

    # Convert to mel spectrogram
    mel_spec = wav_to_mel_spectrogram(wav)
    mel_spec = torch.tensor(mel_spec).unsqueeze(0).to(DEVICE)

    # Process the mel spectrogram through the encoder
    with torch.no_grad():
        audio_embedding = speaker_encoder(mel_spec).cpu().numpy()

    return audio_embedding.squeeze(0)

def extract_description_embeddings_batch(texts):
    """
    Process a batch of text descriptions to extract their embeddings.
    :param texts: List of text description strings.
    :return: the normalized embedding tensor.
    """
    # Use the description encoder to get the embeddings
    with torch.no_grad():
        text_embeddings = description_encoder.encode(texts, convert_to_tensor=True)
        return F.normalize(text_embeddings, p=2, dim=1)


def extract_description_embedding(description):
    """
    Process a single text description to extract its embedding.
    :param description: Text description string.
    :return: the normalized embedding tensor.
    """
    # Use the description encoder to get the embedding
    with torch.no_grad():
        text_embedding = description_encoder.encode([description], convert_to_tensor=True)
        return F.normalize(text_embedding, p=2, dim=1)



def generate_negated_descriptions_batch(texts):
    """Generate negated descriptions for a batch of texts."""
    # Join all descriptions with a separator for batch processing
    batch_prompt = f"""
    Please negate the key characteristics in each of these voice descriptions by changing attributes to their opposites.
    For example, change:
    - gender (male to female, female to male)
    - age (young to old, old to young)
    - pace (fast to slow, slow to fast)
    - pitch (high to low, low to high)
    - accent (British to American, etc.)

    Descriptions:
    """ + '\n'.join(f'{i+1}. "{text}"' for i, text in enumerate(texts)) + """

    Provide your answer as numbered list, with ONLY the negated descriptions (no explanations).
    """
    
    response = ollama.generate(model="llama3.2", prompt=batch_prompt).response
    
    # Parse the response into individual descriptions
    # Split by newlines and filter out empty lines
    lines = [line.strip() for line in response.split('\n') if line.strip()]
    
    # Remove numbering and clean up the descriptions
    descriptions = []
    for i, line in enumerate(lines):
        # Remove leading numbers and dots (e.g., "1. ", "2. ")
        if '. ' in line:
            description = line.split('. ', 1)[1]
            # Remove quotes if present
            description = description.strip('"')
            logger.info(f"generated negated prompt: {description}\n for original text: {texts[i]}")
            descriptions.append(description)
    
    # Ensure we have the same number of descriptions as inputs
    if len(descriptions) != len(texts):
        # If parsing failed, fall back to processing one at a time
        print("Error in generating negative descriptions: Parsing batched response failed.")
        descriptions = [generate_negated_description(text) for text in texts]
    
    return descriptions

def generate_negated_description(text):
    """Generate a negated version of the description using Ollama and Llama3.2"""

    prompt = f"""
    Please negate the key characteristics in this voice description by changing attributes to their opposites.
    For example:
    - Change gender (male to female, female to male)
    - Change age (young to old, old to young)
    - Change pace (fast to slow, slow to fast)
    - Change pitch (high to low, low to high)
    - Change accent (British to American, etc.)

    Original description: "{text}"

    your answer should only include a sentence with the negated characteristics.
    """
    negated_prompt = ollama.generate(model="llama3.2", prompt=prompt).response
    logger.info("generated negated prompt: %s\n for original text: %s", negated_prompt, text)
    return negated_prompt





def refine_description(description, gender, age, speed, pitch, energy, emotion):
    prompt = REFINEMENT_PROMPT_TEMPLATE.format(
        gender=gender,
        age=age,
        speed=speed,
        pitch=pitch,
        energy=energy,
        emotion=emotion,
        original_description=description)

    response = ollama.generate(model="llama3.2", prompt=prompt).response.strip()
    return response.strip('"')


def create_dataset(dataset_size: str = "s"):
    """
    Load the GigaSpeech dataset paired with descriptions from a CSV file.
    The function merges the dataset with the descriptions and saves the result to disk.
    :param dataset_size: Size of the dataset to load (e.g., "s" for small).
    :return: Path to the saved dataset.
    """
    # === STEP 1: Load the CSV file ===
    try:
        df = pd.read_csv(GIGASPEECH_SPEECH_DESCRIPTION_DATASET_PATH)
    except Exception as e:
        print(f"Failed to read the GigaSpeech description file: {e}")
        return "EMPTY_PATH"

    # Validate that the expected columns exist
    if not {ID_COLUMN, DESCRIPTION_COLUMN}.issubset(df.columns):
        print(f"CSV file {GIGASPEECH_SPEECH_DESCRIPTION_DATASET_PATH} must contain columns: {ID_COLUMN}, {DESCRIPTION_COLUMN}")
        return "EMPTY_PATH"

    try:
        # Load the labels dataset
        labels_df = pd.read_csv(GIGASPEECH_SPEECH_LABELS_DATASET_PATH)
    except Exception as e:
        print(f"Failed to read the GigaSpeech labels file: {e}")
        return "EMPTY_PATH"

    if not {ID_COLUMN, GENDER_COLUMN, AGE_COLUMN, SPEED_COLUMN, PITCH_COLUMN, ENERGY_COLUMN, EMOTION_COLUMN}.issubset(labels_df.columns):
        print(f"CSV file {GIGASPEECH_SPEECH_LABELS_DATASET_PATH} must contain columns: {GENDER_COLUMN}, {AGE_COLUMN}, {SPEED_COLUMN}, {PITCH_COLUMN}, {ENERGY_COLUMN}, {EMOTION_COLUMN}")
        return "EMPTY_PATH"


    # Create a dictionary mapping GigaSpeech IDs to descriptions.
    id_to_desc = dict(zip(df[ID_COLUMN], df[DESCRIPTION_COLUMN]))
    id_to_labels = dict(zip(labels_df[ID_COLUMN],
            zip(
                labels_df[GENDER_COLUMN],
                labels_df[AGE_COLUMN],
                labels_df[SPEED_COLUMN],
                labels_df[PITCH_COLUMN],
                labels_df[ENERGY_COLUMN],
                labels_df[EMOTION_COLUMN])))

    print(f"Loaded {len(id_to_desc)} id-description pairs from the CSV file.")
    print(f"Loaded {len(id_to_labels)} id-label pairs from the labels CSV file.")


    # === STEP 2: Load the GigaSpeech dataset ===
    # Validate that the dataset size is one of the valid options

    # Check if the dataset size is valid
    if dataset_size not in VALID_DATASET_SIZES:
        print(f"Invalid dataset size '{dataset_size}'. Valid options are {VALID_DATASET_SIZES}.")
        return "EMPTY_PATH"

    try:
        login(token=HUGGINGFACE_TOKEN, add_to_git_credential=True)
        # Use trust_remote_code=True as the dataset repository calls for it
        giga_dataset = load_dataset("speechcolab/gigaspeech", name=dataset_size, split='train', trust_remote_code=True)
    except Exception as e:
        print(f"Error loading the GigaSpeech dataset: {e}")
        return "EMPTY_PATH"

    # === STEP 3: Merge the datasets by adding descriptions, labels and other data ===
    def add_description_and_labels(example):
        sample_id = example[ID_COLUMN]# Get the sample ID from the example
        text_description =id_to_desc[sample_id] # Get the text description from the id_to_desc dictionary
        labels = id_to_labels[sample_id] # Get the labels from the id_to_labels dictionary

        # add the text description to the example
        example[DESCRIPTION_COLUMN] = text_description
        # add the speaker labels to the example
        example[GENDER_COLUMN] = labels[0]
        example[AGE_COLUMN] = labels[1]
        example[SPEED_COLUMN] = labels[2]
        example[PITCH_COLUMN] = labels[3]
        example[ENERGY_COLUMN] = labels[4]
        example[EMOTION_COLUMN] = labels[5]

        return example

    # Filter the GigaSpeech dataset to only include entries that have a description and labels
    print(f"GigaSpeech dataset has {giga_dataset.num_rows} entries before filtering.")
    giga_dataset = giga_dataset.filter(lambda example: example[ID_COLUMN] in id_to_desc and example[ID_COLUMN] in id_to_labels)
    # Print the number of entries in the GigaSpeech dataset after filtering
    print(f"GigaSpeech dataset has {giga_dataset.num_rows} entries after filtering.")
    # Map the function to add audio and speaker data to each example in the GigaSpeech dataset
    # Note: This will add the audio embedding, text description, negated description, and speaker labels to each example
    print("Adding audio and speaker data to the GigaSpeech dataset...")
    combined_dataset = giga_dataset.map(add_description_and_labels)

    # Optionally keep only the columns you need
    keep_columns = [ID_COLUMN, AUDIO_COLUMN, DESCRIPTION_COLUMN, GENDER_COLUMN, AGE_COLUMN, SPEED_COLUMN, PITCH_COLUMN, ENERGY_COLUMN, EMOTION_COLUMN]
    all_columns = combined_dataset.column_names
    columns_to_remove = [col for col in all_columns if col not in keep_columns]
    if columns_to_remove:
        combined_dataset = combined_dataset.remove_columns(columns_to_remove)

    # === STEP 4: Save the combined dataset ===
    try:
        combined_dataset.save_to_disk(COMBINED_DATASET_PATH)
        print(f"Combined dataset saved to {COMBINED_DATASET_PATH}.")
    except Exception as e:
        print(f"Error saving the dataset to disk: {e}")
        return "EMPTY_PATH"


def enrich_dataset(input_dataset_path = COMBINED_DATASET_PATH, output_dataset_path = ENRICHED_DATASET_PATH, batched=False, batch_size=16,
                   max_dataset_size = 10000):
    """
    Enrich the dataset with speaker embeddings and negated descriptions.
    :param input_dataset_path: the path to the input dataset to enrich.
    :param output_dataset_path: the path to save the enriched dataset.
    :param batched: whether to process the dataset in batches.
    :param batch_size: size of batches for processing.
    :param max_dataset_size: maximum number of entries to process.
    """
    try:
        dataset = load_from_disk(input_dataset_path)
        print(f"Dataset at {input_dataset_path} loaded successfully.")
        if dataset.num_rows > max_dataset_size:
            dataset = dataset.shuffle(seed=213).select(range(max_dataset_size))
    except Exception as e:
        print(f"Error loading the dataset from {input_dataset_path}: {e}")
        return "EMPTY_PATH"

    def process_batch(examples):
        # Generate speaker embeddings for the batch
        audio_batch = examples[AUDIO_COLUMN]
        speaker_embeddings = extract_audio_embeddings_batch(audio_batch)
        
        # Generate negated descriptions for the batch and the descriptions embeddings
        descriptions_batch = examples[DESCRIPTION_COLUMN]
        negated_descriptions = generate_negated_descriptions_batch(descriptions_batch)
        descriptions_embeddings = extract_description_embeddings_batch(descriptions_batch)
        negated_descriptions_embeddings = extract_description_embeddings_batch(negated_descriptions)
        
        # Update the examples with the new data
        examples[RESEMBLYZER_SPEAKER_EMBEDDING_COLUMN] = speaker_embeddings
        examples[NEGATIVE_DESCRIPTION_COLUMN] = negated_descriptions
        examples[GRANITE_DESCRIPTION_EMBEDDING_COLUMN] = descriptions_embeddings
        examples[GRANITE_NEGATIVE_DESCRIPTION_EMBEDDING_COLUMN] = negated_descriptions_embeddings
        
        return examples

    def process_single(example):
        # Generate speaker embedding for a single example
        audio = example[AUDIO_COLUMN]
        speaker_embedding = extract_audio_embedding(audio)

        # Generate negated description and its embedding
        description = example[DESCRIPTION_COLUMN]
        negated_description = generate_negated_description(description)
        description_embedding = extract_description_embedding(description)
        negated_description_embedding = extract_description_embedding(negated_description)

        # Update the example with the new data
        example[RESEMBLYZER_SPEAKER_EMBEDDING_COLUMN] = speaker_embedding
        example[NEGATIVE_DESCRIPTION_COLUMN] = negated_description
        example[GRANITE_DESCRIPTION_EMBEDDING_COLUMN] = description_embedding
        example[GRANITE_NEGATIVE_DESCRIPTION_EMBEDDING_COLUMN] = negated_description_embedding

        return example

    print("Enriching dataset in batches..." if batched else "Enriching dataset one example at a time...")
    enriched_dataset = dataset.map(
        process_batch if batched else process_single,
        batched=batched,
        batch_size=batch_size,
        desc="Processing batches" if batched else "Processing single examples",
    )

    split = enriched_dataset.train_test_split(test_size=0.1, seed=213)
    train_dataset = split["train"]
    test_dataset = split["test"]
    full_enriched_dataset = DatasetDict({'train': train_dataset, 'test': test_dataset})

    # Save the enriched dataset
    full_enriched_dataset.save_to_disk(output_dataset_path)
    print(f"Enriched dataset saved to {output_dataset_path}.")


def enrich_dataset_v2(input_dataset_path=COMBINED_DATASET_PATH, output_dataset_path=ENRICHED_DATASET_V2_PATH, batched=False, batch_size=16,
                     max_dataset_size=10000):
    """
    Enrich the dataset with speaker embeddings and negated descriptions.
    :param input_dataset_path: the path to the input dataset to enrich.
    :param output_dataset_path: the path to save the enriched dataset.
    :param batched: whether to process the dataset in batches.
    :param batch_size: size of batches for processing.
    :param max_dataset_size: maximum number of entries to process.
    """

    try:
        dataset = load_from_disk(input_dataset_path)
        print(f"Dataset at {input_dataset_path} loaded successfully.")
        if dataset.num_rows > max_dataset_size:
            dataset = dataset.shuffle(seed=213).select(range(max_dataset_size))
    except Exception as e:
        print(f"Error loading the dataset from {input_dataset_path}: {e}")
        return "EMPTY_PATH"

    def process_batch(examples):
        # Generate speaker embeddings for the batch
        audio_batch = examples[AUDIO_COLUMN]
        speaker_embeddings = extract_audio_embeddings_batch(audio_batch)

        # Generate negated descriptions for the batch and the descriptions embeddings
        descriptions_batch = [refine_description(examples[DESCRIPTION_COLUMN][i],
                                                 examples[GENDER_COLUMN][i],
                                                    examples[AGE_COLUMN][i],
                                                    examples[SPEED_COLUMN][i],
                                                    examples[PITCH_COLUMN][i],
                                                    examples[ENERGY_COLUMN][i],
                                                    examples[EMOTION_COLUMN][i]) for i in range(len(examples))]
        negated_descriptions = generate_negated_descriptions_batch(descriptions_batch)
        descriptions_embeddings = extract_description_embeddings_batch(descriptions_batch)
        negated_descriptions_embeddings = extract_description_embeddings_batch(negated_descriptions)

        # Update the examples with the new data
        examples[RESEMBLYZER_SPEAKER_EMBEDDING_COLUMN] = speaker_embeddings
        examples[NEGATIVE_DESCRIPTION_COLUMN] = negated_descriptions
        examples[GRANITE_DESCRIPTION_EMBEDDING_COLUMN] = descriptions_embeddings
        examples[GRANITE_NEGATIVE_DESCRIPTION_EMBEDDING_COLUMN] = negated_descriptions_embeddings
        examples[DESCRIPTION_COLUMN] = descriptions_batch

        return examples

    def process_single(example):
        # Generate speaker embedding for a single example
        audio = example[AUDIO_COLUMN]
        speaker_embedding = extract_audio_embedding(audio)

        # Generate negated description and its embedding
        description = refine_description(example[DESCRIPTION_COLUMN],
                                         example[GENDER_COLUMN],
                                            example[AGE_COLUMN],
                                            example[SPEED_COLUMN],
                                            example[PITCH_COLUMN],
                                            example[ENERGY_COLUMN],
                                            example[EMOTION_COLUMN])

        negated_description = generate_negated_description(description)
        description_embedding = extract_description_embedding(description)
        negated_description_embedding = extract_description_embedding(negated_description)

        # Update the example with the new data
        example[RESEMBLYZER_SPEAKER_EMBEDDING_COLUMN] = speaker_embedding
        example[NEGATIVE_DESCRIPTION_COLUMN] = negated_description
        example[GRANITE_DESCRIPTION_EMBEDDING_COLUMN] = description_embedding
        example[GRANITE_NEGATIVE_DESCRIPTION_EMBEDDING_COLUMN] = negated_description_embedding
        example[DESCRIPTION_COLUMN] = description

        return example

    print("Enriching dataset in batches..." if batched else "Enriching dataset in single examples...")
    enriched_dataset = dataset.map(
        process_batch if batched else process_single,
        batched=batched,
        batch_size=batch_size,
        desc="Processing batches" if batched else "Processing single examples",
    )

    split = enriched_dataset.train_test_split(test_size=0.1, seed=213)
    train_dataset = split["train"]
    test_dataset = split["test"]
    full_enriched_dataset = DatasetDict({"train": train_dataset, "test": test_dataset})

    # Save the enriched dataset
    full_enriched_dataset.save_to_disk(output_dataset_path)
    print(f"Enriched dataset saved to {output_dataset_path}.")

from datasets import load_from_disk, Dataset
from resemblyzer import VoiceEncoder, preprocess_wav

DATASET_PATH = "../datasets/CoTTS_dataset"
EMPTY_DATASET = Dataset.from_dict({"segment_id": [], "audio": [], "description": []})


def resemblyzer_audio_encoder(example: dict) -> dict:
    """
    Encode audio using the VoiceEncoder from resemblyzer.
    :param example: A dictionary containing the audio data.
    :return: The input dictionary with an additional key for the audio embedding.
    """
    encoder = VoiceEncoder()
    try:
        wav = preprocess_wav(example["audio"]["path"])
        encoder = VoiceEncoder()
        embedding = encoder.embed_speaker([wav])
        example["audio_embedding"] = embedding
    except KeyError as e:
        print(f"KeyError: path not found in audio dict: {e}")
    except Exception as e:
        print(f"Error processing audio file in example {example['segment_id']}: {e}")
    return example



def create_encoded_audio_dataset(dataset_path: str = DATASET_PATH) -> Dataset:
    """
    Load a dataset from disk, preprocess the audio files, and encode them using VoiceEncoder.
    :param dataset_path: Path to the dataset.
    :return: A Dataset object with audio embeddings.
    """

    try:
        dataset = load_from_disk(dataset_path)
        print(f"Successfully loaded dataset with {len(dataset)} samples.")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return EMPTY_DATASET

    # Check if the dataset has the expected columns
    expected_columns = ["segment_id", "audio", "description"]
    if not all(col in dataset.column_names for col in expected_columns):
        print(f"Dataset must contain columns: {expected_columns}")
        return EMPTY_DATASET

    # Apply the encoding function to the dataset
    try:
        dataset = dataset.map(encode_audio, remove_columns=["audio"], num_proc=4)
        print("Audio encoding completed.")
        return dataset
    except Exception as e:
        print(f"Error during mapping: {e}")
        return EMPTY_DATASET






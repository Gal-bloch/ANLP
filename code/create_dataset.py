import os

# Ensure Python uses UTF-8 as default if possible
os.environ["PYTHONUTF8"] = "1"


# === Updated Monkey-patch builtins.open for read mode (non-binary) ===
import builtins

# Save the original open
original_open = builtins.open


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
from datasets import load_dataset
from huggingface_hub import login

# Constants for your description CSV file.
GIGASPEECH_SPEECH_DESCRIPTION_DATASET_PATH = "../datasets/EN_description_GIGASPEECH.csv"
ID_COLUMN = "segment_id"
DESCRIPTION_COLUMN = "description"

COMBINED_DATASET_PATH = "../datasets/CoTTS_dataset"

VALID_DATASET_SIZES = ["xs", "s", "m", "l", "xl"]

HUGGINGFACE_TOKEN = "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"


def create_dataset(dataset_size: str = "s",
                         description_dataset_path: str = GIGASPEECH_SPEECH_DESCRIPTION_DATASET_PATH,
                         id_column: str = ID_COLUMN, description_column: str = DESCRIPTION_COLUMN,
                         huggingface_token: str = HUGGINGFACE_TOKEN, save_path: str = COMBINED_DATASET_PATH) -> str:
    """
    Load the GigaSpeech dataset paired with descriptions from a CSV file.
    The function merges the dataset with the descriptions and saves the result to disk.
    :param dataset_size: Size of the dataset to load (e.g., "s" for small).
    :param description_dataset_path: Path to the CSV file containing descriptions.
    :param id_column: Column name in the CSV file for the segment IDs.
    :param description_column: Column name in the CSV file for the descriptions.
    :param huggingface_token: Hugging Face token for authentication.
    :param save_path: Path where the combined dataset will be saved.
    :return: Path to the saved dataset.
    """
    # === STEP 1: Load the CSV file ===
    try:
        df = pd.read_csv(description_dataset_path)
    except Exception as e:
        print(f"Failed to read the GigaSpeech description file: {e}")
        return "EMPTY_PATH"

    # Validate that the expected columns exist
    if not {id_column, description_column}.issubset(df.columns):
        print(f"CSV file must contain columns '{ID_COLUMN}' and '{DESCRIPTION_COLUMN}'.")
        return "EMPTY_PATH"

    # Create a dictionary mapping GigaSpeech IDs to descriptions.
    id_to_desc = dict(zip(df[ID_COLUMN], df[DESCRIPTION_COLUMN]))
    print(f"Loaded {len(id_to_desc)} id-description pairs from the CSV file.")


    # === STEP 2: Load the GigaSpeech dataset ===
    # Validate that the dataset size is one of the valid options

    # Check if the dataset size is valid
    if dataset_size not in VALID_DATASET_SIZES:
        print(f"Invalid dataset size '{dataset_size}'. Valid options are {VALID_DATASET_SIZES}.")
        return "EMPTY_PATH"

    try:
        login(token=huggingface_token, add_to_git_credential=True)
        # Use trust_remote_code=True as the dataset repository calls for it
        giga_dataset = load_dataset("speechcolab/gigaspeech", name=dataset_size, split="train", trust_remote_code=True)
    except Exception as e:
        print(f"Error loading the GigaSpeech dataset: {e}")
        return "EMPTY_PATH"

    # === STEP 3: Merge the datasets by adding descriptions ===
    def add_description(example):
        sample_id = example.get("segment_id", None)
        if sample_id in id_to_desc:
            example["description"] = id_to_desc[sample_id]
        else:
            example["description"] = ""
        return example

    giga_dataset = giga_dataset.map(add_description)
    combined_dataset = giga_dataset.filter(lambda example: example["description"] != "")
    print(f"Combined dataset has {len(combined_dataset)} entries after filtering.")

    # Optionally keep only the columns you need
    keep_columns = ["segment_id", "audio", "description"]
    all_columns = combined_dataset.column_names
    columns_to_remove = [col for col in all_columns if col not in keep_columns]
    if columns_to_remove:
        combined_dataset = combined_dataset.remove_columns(columns_to_remove)

    # === STEP 4: Save the combined dataset ===
    try:
        combined_dataset.save_to_disk(save_path)
        print(f"Combined dataset saved to {save_path}.")
    except Exception as e:
        print(f"Error saving the dataset to disk: {e}")
        return "EMPTY_PATH"

    return save_path


if __name__ == "__main__":
    create_dataset(dataset_size='m')



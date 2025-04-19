from itertools import islice

from voice_to_embedding import Voice2Embedding
from sentence_transformers import SentenceTransformer
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from resemblyzer import VoiceEncoder, wav_to_mel_spectrogram
import torchaudio
import librosa
import numpy as np
import os
import logging
from datasets import load_dataset, concatenate_datasets, load_from_disk, Audio
from tqdm import tqdm
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Set directory for downloaded audio
AUDIO_CACHE_DIR = "./audio_cache"
os.makedirs(AUDIO_CACHE_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,  # Set to DEBUG for more detailed logs
    format="%(asctime)s [%(levelname)s]: %(message)s",
    handlers=[
        logging.StreamHandler()  # Print to console
    ]
)

logger = logging.getLogger(__name__)


# def merge_datasets_with_audio(id_column="original_path"):
#     """
#     Merges English audio dataset with its metadata while keeping audio files.
#
#     Args:
#         id_column (str): Column used to align both datasets.
#
#     Returns:
#         datasets.DatasetDict: Merged dataset with original structure + metadata.
#     """
#     dataset_name = "parler-tts/mls_eng"
#     metadata_dataset_name = "parler-tts/mls-eng-speaker-descriptions"
#
#     print(f"Loading audio dataset: {dataset_name}")
#     # Load as streaming but convert to a dictionary of limited datasets
#     loader = load_dataset(dataset_name, streaming=True)
#
#     # Create a dict to hold limited data by split
#     dataset = {}
#     for split_name, max_examples in zip(['test', 'dev', 'train'], [100, 50, 2500]):
#         logger.info(f"Taking first {max_examples} examples from '{split_name}' split")
#         split_data = list(islice(loader[split_name], max_examples))
#         # Convert the list of examples to a dataset
#         from datasets import Dataset
#         dataset[split_name] = Dataset.from_list(split_data)
#
#     logger.info(f"Loading metadata dataset: {metadata_dataset_name}")
#     metadata_dataset = load_dataset(metadata_dataset_name)
#
#     merged_data = {}
#
#     for split in dataset:
#         if split in metadata_dataset:
#             logger.info(f"Merging '{split}' split...")
#
#             # Limit metadata to match the audio dataset size
#             metadata_split = metadata_dataset[split].select(
#                 range(min(len(dataset[split]), len(metadata_dataset[split]))))
#
#             # Get the list of columns from both datasets
#             audio_columns = dataset[split].column_names
#             metadata_columns = metadata_split.column_names
#
#             # Find common columns that need to be renamed in metadata dataset
#             duplicated_columns = [col for col in metadata_columns if col in audio_columns and col != id_column]
#
#             # Rename the duplicated columns in metadata dataset
#             for col in duplicated_columns:
#                 metadata_split = metadata_split.rename_column(col, f"metadata_{col}")
#
#             # Rename the ID column to make validation easier
#             metadata_split = metadata_split.rename_column(id_column, f"metadata_{id_column}")
#
#             # Concatenate datasets
#             merged_split = concatenate_datasets([dataset[split], metadata_split], axis=1)
#
#             # Validate alignment
#             if len(merged_split.filter(lambda id1, id2: id1 != id2,
#                                        input_columns=[id_column, f"metadata_{id_column}"])) != 0:
#                 raise ValueError(f"Mismatch in IDs after merging on split {split}")
#
#             merged_data[split] = merged_split
#         else:
#             logger.warning(f"Split '{split}' is missing from the metadata dataset.")
#
#     logger.info("✅ Dataset successfully merged!")
#     return merged_data
#
# class MLSSpeakerDataset(Dataset):
#     def __init__(self, hf_dataset, split='train', max_samples=None):
#         """
#         Custom dataset class that loads audio and metadata.
#
#         Args:
#             hf_dataset (datasets.DatasetDict): Hugging Face dataset with merged audio and metadata.
#             max_samples (int, optional): Maximum number of samples to use.
#         """
#         self.dataset = hf_dataset[split]
#         if max_samples:
#             self.dataset = self.dataset.select(range(min(max_samples, len(self.dataset))))
#
#     def __len__(self):
#         return len(self.dataset)
#
#     def __getitem__(self, idx):
#         item = self.dataset[idx]
#         description = item["text_description"]
#         description = " ".join(description) if isinstance(description, list) else description
#
#         audio_path = item["audio"]["path"]  # Use the local file path
#         print(audio_path)
#
#         if not audio_path or not os.path.exists(audio_path):
#             return np.zeros(16000), description  # Handle missing files
#
#         try:
#             waveform, sample_rate = torchaudio.load(audio_path)
#             wav = waveform.mean(dim=0).numpy()
#         except Exception:
#             try:
#                 wav, sample_rate = librosa.load(audio_path, sr=None, mono=True)
#             except Exception:
#                 return np.zeros(16000), description  # Handle errors
#
#         return wav, description

def get_text_embedding(texts, model):
    """Get text embeddings using SentenceTransformer model"""
    with torch.no_grad():
        embeddings = model.encode(texts, convert_to_tensor=True)
        return F.normalize(embeddings, p=2, dim=1)  # Normalize embeddings


def cosine_similarity_loss(speech_emb, text_emb):
    """Loss function that maximizes cosine similarity"""
    return 1 - F.cosine_similarity(speech_emb, text_emb, dim=1).mean()


def pad_sequence(sequences, batch_first=False, padding_value=0.0):
    """
    Pads a batch of sequences to the same length by padding with zeros (or other specified value),
    for both time and frequency dimensions (height and width of spectrogram).

    Args:
        sequences (list): List of tensors of variable length.
        batch_first (bool): Whether to return the padded sequence as (batch_size, seq_len) or (seq_len, batch_size).
        padding_value (float): Value used for padding.

    Returns:
        Tensor: Padded sequence.
    """
    # Find the maximum time and frequency lengths across all sequences
    max_time_len = max([seq.size(2) for seq in sequences])  # Max length in time dimension (width)
    max_freq_len = max([seq.size(1) for seq in sequences])  # Max length in frequency dimension (height)

    # Pad each sequence to match the max lengths
    padded_sequences = []
    for seq in sequences:
        pad_time = max_time_len - seq.size(2)  # Padding for time dimension
        pad_freq = max_freq_len - seq.size(1)  # Padding for frequency dimension

        # Pad along both dimensions (frequency and time)
        padded_seq = F.pad(seq, (0, pad_time, 0, pad_freq), value=padding_value)  # Pad (time, frequency)
        padded_sequences.append(padded_seq)

    # Stack all padded sequences into a single tensor
    padded_tensor = torch.stack(padded_sequences, dim=0)

    if batch_first:
        return padded_tensor
    else:
        return padded_tensor.transpose(0, 1)


def collate_fn(batch):
    """Custom collate function to handle variable length audio and process them for the VoiceEncoder"""
    wavs = [item["audio"]["array"] for item in batch]
    descriptions = [item["text_description"] for item in batch]

    # Process each audio to the format expected by VoiceEncoder
    # VoiceEncoder expects mel spectrograms in a specific format
    mel_specs = []
    for wav in wavs:
        # Convert to mel spectrogram according to the requirements of the VoiceEncoder
        mel_spec = wav_to_mel_spectrogram(wav)
        mel_specs.append(torch.tensor(mel_spec))

    return mel_specs, descriptions


def main():
    # logger.info("Merging datasets...")
    # dataset = merge_datasets_with_audio()
    #
    # logger.info("Preparing training and validation datasets")
    # train_dataset = MLSSpeakerDataset(dataset, 'train', max_samples=2500)
    # val_dataset = MLSSpeakerDataset(dataset, 'dev', max_samples=50)

    dataset = load_from_disk("/Users/galbloch/Desktop/school/git/ANLP/datasets/CoTTS_dataset")
    dataset = dataset.cast_column("audio", Audio())

    # Rename + clean
    dataset = dataset.rename_column("description", "text_description").remove_columns(["segment_id"])

    # Filter down to a max number of samples (if needed)
    max_samples = 1000
    if max_samples < len(dataset):
        dataset = dataset.select(range(max_samples))

    # Split into train and validation (e.g., 90/10 split)
    split_dataset = dataset.train_test_split(test_size=0.1, seed=42)  # 10% for validation
    train_dataset = split_dataset["train"]
    val_dataset = split_dataset["test"]

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn, num_workers=4)

    logger.info("Initializing model")
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    dense_embedding_model = SentenceTransformer("ibm-granite/granite-embedding-125m-english")
    voice_encoder = VoiceEncoder(device=device)
    model = Voice2Embedding(voice_encoder, projection_dim=dense_embedding_model.get_sentence_embedding_dimension())
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.3, patience=5)

    logger.info("Running training loop")
    num_epochs = 50
    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for mel_specs, descriptions in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} - Training"):
            # Process each mel spectrogram in the batch
            speech_embeddings = []
            for mel_spec in mel_specs:
                # Move to device and ensure correct shape
                mel_spec = mel_spec.to(device)

                # REMOVE the torch.no_grad() here - we need gradients for training
                emb = model(mel_spec.unsqueeze(0))  # Add batch dimension
                speech_embeddings.append(emb)

            # Stack all embeddings
            speech_emb = torch.cat(speech_embeddings, dim=0)

            # Get text embeddings
            text_emb = get_text_embedding(descriptions, dense_embedding_model).to(device)

            # Calculate loss
            loss = cosine_similarity_loss(speech_emb, text_emb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # Validation - similar approach
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for mel_specs, descriptions in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} - Validation"):
                # Process each mel spectrogram in the batch
                speech_embeddings = []
                for mel_spec in mel_specs:
                    mel_spec = mel_spec.to(device)
                    emb = model(mel_spec.unsqueeze(0))
                    speech_embeddings.append(emb)

                speech_emb = torch.cat(speech_embeddings, dim=0)
                text_emb = get_text_embedding(descriptions, dense_embedding_model).to(device)

                loss = cosine_similarity_loss(speech_emb, text_emb)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)

        logger.info(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(
                {"epoch": epoch, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                 "val_loss": best_val_loss},
                "best_voice2embedding_model.pt",
            )
            logger.info(f"Saved new best model with validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()

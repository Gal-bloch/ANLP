from voice_to_embedding import Voice2Embedding
from sentence_transformers import SentenceTransformer
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from resemblyzer import VoiceEncoder, wav_to_mel_spectrogram
import logging
from datasets import load_from_disk, Audio
from tqdm import tqdm
import os
from create_dataset import ENRICHED_DATASET_PATH, AUDIO_COLUMN, SPEAKER_EMBEDDING_COLUMN, DESCRIPTION_EMBEDDING_COLUMN, NEGATIVE_DESCRIPTION_EMBEDDING_COLUMN

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


def cosine_similarity_loss(speech_emb, text_emb):
    """Loss function that maximizes cosine similarity"""
    return 1 - F.cosine_similarity(speech_emb, text_emb, dim=1).mean()


# Min-max normalization for loss computation
def min_max_normalize(tensor):
    """Apply min-max normalization to tensor values to range [0, 1]"""
    min_val = torch.min(tensor)
    max_val = torch.max(tensor)
    if min_val == max_val:
        return torch.zeros_like(tensor)
    return (tensor - min_val) / (max_val - min_val)


def contrastive_loss(speech_emb, text_emb, neg_text_emb, margin=0.5):
    """
    Contrastive loss using min-max normalization:
    - Maximize similarity between speech and positive text
    - Minimize similarity between speech and negative text
    """
    # Positive similarity (speech and matching text)
    pos_sim = F.cosine_similarity(speech_emb, text_emb, dim=1)

    # Negative similarity (speech and negated text)
    neg_sim = F.cosine_similarity(speech_emb, neg_text_emb, dim=1)

    # Normalize similarities
    pos_sim_norm = min_max_normalize(pos_sim)
    neg_sim_norm = min_max_normalize(neg_sim)

    # Loss: minimize positive distance, maximize negative distance
    loss = (1 - pos_sim_norm) + torch.clamp(neg_sim_norm - margin, min=0)

    return loss.mean()

def info_nce_loss(speech_emb, text_emb, temperature=0.07):
    """
    InfoNCE loss with in-batch negatives (cross-modal)
    speech_emb: (B, D)
    text_emb: (B, D)
    """
    # Normalize
    speech_emb = F.normalize(speech_emb, p=2, dim=1)
    text_emb = F.normalize(text_emb, p=2, dim=1)

    logits = torch.matmul(speech_emb, text_emb.T) / temperature  # (B, B)
    labels = torch.arange(len(speech_emb)).to(speech_emb.device)  # Ground truth is the diagonal

    loss_i2t = F.cross_entropy(logits, labels)       # Speech → Text
    loss_t2i = F.cross_entropy(logits.T, labels)     # Text → Speech

    return (loss_i2t + loss_t2i) / 2


def collate_fn(batch):
    """Custom collate function using pre-generated negated descriptions"""
    wavs = [item[AUDIO_COLUMN]["array"] for item in batch]
    descriptions_embeddings = torch.tensor([item[DESCRIPTION_EMBEDDING_COLUMN] for item in batch])
    negated_descriptions_embeddings = torch.tensor([item[NEGATIVE_DESCRIPTION_EMBEDDING_COLUMN] for item in batch])

    mel_specs = [torch.tensor(wav_to_mel_spectrogram(wav)) for wav in wavs]

    return mel_specs, descriptions_embeddings, negated_descriptions_embeddings


def main():
    # logger.info("Merging datasets...")
    # dataset = merge_datasets_with_audio()
    #
    # logger.info("Preparing training and validation datasets")
    # train_dataset = MLSSpeakerDataset(dataset, 'train', max_samples=2500)
    # val_dataset = MLSSpeakerDataset(dataset, 'dev', max_samples=50)

    dataset = load_from_disk(ENRICHED_DATASET_PATH)
    dataset = dataset.cast_column("audio", Audio())

    # Split into train and validation (e.g., 90/10 split)
    split_dataset = dataset.train_test_split(test_size=0.1, seed=42)  # 10% for validation
    train_dataset = split_dataset["train"]
    val_dataset = split_dataset["test"]

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn, num_workers=4)

    logger.info("Initializing model")
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
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

        for mel_specs, desc_emb, neg_desc_emb in tqdm(train_loader,
                                                                  desc=f"Epoch {epoch + 1}/{num_epochs} - Training"):
            speech_embeddings = []
            for mel_spec in mel_specs:
                mel_spec = mel_spec.to(device)
                emb = model(mel_spec.unsqueeze(0))
                speech_embeddings.append(emb)

            speech_emb = torch.cat(speech_embeddings, dim=0).to(device)
            desc_emb = desc_emb.to(device)
            neg_desc_emb = neg_desc_emb.to(device)

            loss = contrastive_loss(speech_emb, desc_emb, neg_desc_emb)



            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # Validation - now with negated descriptions too
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for mel_specs, desc_emb, neg_desc_emb in tqdm(val_loader,
                                                                      desc=f"Epoch {epoch + 1}/{num_epochs} - Validation"):
                speech_embeddings = []
                for mel_spec in mel_specs:
                    mel_spec = mel_spec.to(device)
                    emb = model(mel_spec.unsqueeze(0))
                    speech_embeddings.append(emb)

                speech_emb = torch.cat(speech_embeddings, dim=0).to(device)
                desc_emb = desc_emb.to(device)
                neg_desc_emb = neg_desc_emb.to(device)

                loss = contrastive_loss(speech_emb, desc_emb, neg_desc_emb)
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
                "../models/best_voice2embedding_model.pt",
            )
            logger.info(f"Saved new best model with validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()

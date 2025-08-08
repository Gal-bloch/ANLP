import torch.nn as nn
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader
from resemblyzer import VoiceEncoder, wav_to_mel_spectrogram
import logging
from datasets import load_from_disk, Audio
from tqdm import tqdm
import os
from create_dataset import ENRICHED_DATASET_V2_PATH, AUDIO_COLUMN, GRANITE_DESCRIPTION_EMBEDDING_COLUMN, GRANITE_NEGATIVE_DESCRIPTION_EMBEDDING_COLUMN
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



# ========== CONFIG ==========
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
VOICE2EMBEDDING_VOICE_EMBEDDER = VoiceEncoder(device=DEVICE)
VOICE2EMBEDDING_DESCRIPTION_EMBEDDER = SentenceTransformer("ibm-granite/granite-embedding-125m-english")

BATCH_SIZE = 16
NUM_EPOCHS = 50
LR = 1e-5
WEIGHT_DECAY = 1e-4
HEAD_NUM = 8
LAYER_NUM = 3
DROPOUT_RATE = 0.1
TEXT_EMBED_DIM = 768
AUDIO_EMBED_DIM = 256  # VoiceEncoder outputs 256-dim embeddings
VOICE2EMBEDDING_MODEL_PATH = "../models/voice2embedding_model.pt"




# ========== Model Definition ==========

class Voice2Embedding(nn.Module):
    def __init__(self, voice_encoder: nn.Module = VOICE2EMBEDDING_VOICE_EMBEDDER, projection_dim: int = TEXT_EMBED_DIM, nhead: int = HEAD_NUM, num_layers: int = LAYER_NUM,
                 dropout: float = DROPOUT_RATE):
        super().__init__()
        self.base_model = voice_encoder
        self.input_dim = voice_encoder.linear.out_features

        # Normal transformer without potentially unstable components
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.input_dim,
            nhead=nhead,
            dim_feedforward=4 * self.input_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Simple projection head to avoid instability
        self.projection = nn.Sequential(
            nn.Linear(self.input_dim, 2 * projection_dim),
            nn.ReLU(),  # Back to ReLU for stability
            nn.Dropout(dropout),
            nn.Linear(2 * projection_dim, projection_dim)
        )

        # Gradient clipping in init to prevent explosion
        for p in self.parameters():
            if p.requires_grad:
                p.register_hook(lambda grad: torch.clamp(grad, -1.0, 1.0))

        logger.info(f"Number of parameters in base model: {sum(p.numel() for p in self.base_model.parameters())}")
        logger.info(f"Number of parameters in new model: {sum(p.numel() for p in self.parameters())}")

    def encode_speech(self, speech_embeddings):
        # Handle different input dimensions
        if speech_embeddings.dim() == 1:
            speech_embeddings = speech_embeddings.unsqueeze(0)
        if speech_embeddings.dim() == 2:
            speech_embeddings = speech_embeddings.unsqueeze(1)

        # Add small epsilon to prevent zeros
        speech_embeddings = speech_embeddings + 1e-8

        # Apply transformer with stability check
        encoded = self.transformer(speech_embeddings)

        # Simple mean pooling is more stable
        pooled = encoded.mean(dim=1)

        # Check for NaNs and replace if needed
        if torch.isnan(pooled).any():
            pooled = torch.where(torch.isnan(pooled), torch.ones_like(pooled) * 1e-8, pooled)

        projected = self.projection(pooled)

        # Softly normalize to avoid division by zero
        norm = torch.norm(projected, p=2, dim=1, keepdim=True).clamp(min=1e-8)
        normalized = projected / norm

        return normalized

    def forward(self, speech_mel_specs):
        with torch.no_grad():  # Don't backprop through base model initially
            speech_embedding = self.base_model(speech_mel_specs)

        return self.encode_speech(speech_embedding)

    @staticmethod
    def loss(speech_encodings, description_embeddings):
        return 1 - F.cosine_similarity(speech_encodings, description_embeddings, dim=1).mean()

    @staticmethod
    def contrastive_loss(speech_encodings, positive_references, negative_references):
        if type(negative_references) == str:
            negative_references = VOICE2EMBEDDING_DESCRIPTION_EMBEDDER(negative_references)
        if type(positive_references) == str:
            positive_references = VOICE2EMBEDDING_DESCRIPTION_EMBEDDER(positive_references)

        pos_distances = 1 - F.cosine_similarity(speech_encodings, positive_references, dim=1)
        neg_distances = 1 - F.cosine_similarity(speech_encodings, negative_references, dim=1)

        return (pos_distances - neg_distances).mean()





# ========== Training ==========

def collate_fn(batch):
    """Custom collate function using pre-generated negated descriptions"""
    wavs = [item[AUDIO_COLUMN]["array"] for item in batch]
    descriptions_embeddings = torch.tensor([item[GRANITE_DESCRIPTION_EMBEDDING_COLUMN] for item in batch])
    negated_descriptions_embeddings = torch.tensor([item[GRANITE_NEGATIVE_DESCRIPTION_EMBEDDING_COLUMN] for item in batch])

    mel_specs = [torch.tensor(wav_to_mel_spectrogram(wav)) for wav in wavs]

    return mel_specs, descriptions_embeddings, negated_descriptions_embeddings


def main():
    dataset_dict = load_from_disk(ENRICHED_DATASET_V2_PATH)
    train_dataset = dataset_dict["train"].cast_column(AUDIO_COLUMN, Audio())
    test_dataset = dataset_dict["test"].cast_column(AUDIO_COLUMN, Audio())

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)

    logger.info("Initializing model")
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    dense_embedding_model = SentenceTransformer("ibm-granite/granite-embedding-125m-english")
    voice_encoder = VoiceEncoder(device=device)
    model = Voice2Embedding(voice_encoder, projection_dim=dense_embedding_model.get_sentence_embedding_dimension())
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.3, patience=5)

    logger.info("Running training loop")
    num_epochs = 50
    best_val_loss = float("inf")

    train_losses, val_losses = [], []
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

            loss = model.contrastive_loss(speech_emb, desc_emb, neg_desc_emb)

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

                loss = model.contrastive_loss(speech_emb, desc_emb, neg_desc_emb)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        logger.info(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(
                {"epoch": epoch, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                 "val_loss": best_val_loss},
                VOICE2EMBEDDING_MODEL_PATH,
            )
            logger.info(f"Saved new best model with validation loss: {best_val_loss:.4f}")

    model_data = torch.load(VOICE2EMBEDDING_MODEL_PATH)
    model_data["training_losses"] = train_losses
    model_data["validation_losses"] = val_losses
    torch.save(model_data, VOICE2EMBEDDING_MODEL_PATH)

    logger.info("Training complete. Best model saved.")


if __name__ == "__main__":
    main()

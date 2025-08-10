import torch
import torch.nn as nn
import torch.utils.tensorboard.summary
from torch.utils.data import DataLoader
from datasets import load_from_disk, Audio
from tqdm import tqdm
from create_dataset import ENRICHED_DATASET_V2_PATH, AUDIO_COLUMN, RESEMBLYZER_SPEAKER_EMBEDDING_COLUMN, GRANITE_DESCRIPTION_EMBEDDING_COLUMN, GRANITE_NEGATIVE_DESCRIPTION_EMBEDDING_COLUMN
from resemblyzer import VoiceEncoder
from sentence_transformers import SentenceTransformer


# ========== CONFIG ==========
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
DCCA_V2_VOICE_EMBEDDER = VoiceEncoder(device=DEVICE)
DCCA_V2_DESCRIPTION_EMBEDDER = SentenceTransformer("ibm-granite/granite-embedding-125m-english")


BATCH_SIZE = 16
NUM_EPOCHS = 50
LR = 1e-5
WEIGHT_DECAY = 1e-4
DROPOUT_RATE = 0.1
HEAD_NUM = 8
LAYER_NUM = 3
EPS = 1e-3
TEXT_EMBED_DIM = 768
AUDIO_EMBED_DIM = 256  # VoiceEncoder outputs 256-dim embeddings
SHARED_DIM = 128  # Dimension of the shared space
DCCA_V2_MODEL_PATH = "../models/dccav2_voice_text.pt"


# ========== Model Definition ==========

class DCCAV2SpeechText(nn.Module):
    def __init__(self, audio_dim=AUDIO_EMBED_DIM, text_dim=TEXT_EMBED_DIM, shared_dim=SHARED_DIM,
                 dropout_rate=DROPOUT_RATE, nhead=HEAD_NUM, num_layers=LAYER_NUM,
                 device=DEVICE):
        super(DCCAV2SpeechText, self).__init__()
        self.device = device
        self.shared_dim = shared_dim

        # Normal transformer without potentially unstable components
        audio_encoder_layer = nn.TransformerEncoderLayer(
            d_model=audio_dim,
            nhead=nhead,
            dim_feedforward=4 * audio_dim,
            dropout=dropout_rate,
            batch_first=True
        )
        self.audio_transformer = nn.TransformerEncoder(audio_encoder_layer, num_layers=num_layers)

        # Simple projection head to avoid instability
        self.audio_projection = nn.Sequential(
            nn.Linear(audio_dim, 2 * shared_dim),
            nn.ReLU(),  # Back to ReLU for stability
            nn.Dropout(dropout_rate),
            nn.Linear(2 * shared_dim, shared_dim)
        )

        # Normal transformer without potentially unstable components
        text_encoder_layer = nn.TransformerEncoderLayer(
            d_model=text_dim,
            nhead=nhead,
            dim_feedforward=4 * text_dim,
            dropout=dropout_rate,
            batch_first=True
        )
        self.text_transformer = nn.TransformerEncoder(text_encoder_layer, num_layers=num_layers)

        # Simple projection head to avoid instability
        self.text_projection = nn.Sequential(
            nn.Linear(text_dim, 2 * shared_dim),
            nn.ReLU(),  # Back to ReLU for stability
            nn.Dropout(dropout_rate),
            nn.Linear(2 * shared_dim, shared_dim)
        )

        # Gradient clipping in init to prevent explosion
        for p in self.parameters():
            if p.requires_grad:
                p.register_hook(lambda grad: torch.clamp(grad, -1.0, 1.0))

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def encode_speech(self, audio_embs):
        # Handle different input dimensions
        if audio_embs.dim() == 1:
            audio_embs = audio_embs.unsqueeze(0)
        if audio_embs.dim() == 2:
            audio_embs = audio_embs.unsqueeze(1)

        # Add small epsilon to prevent zeros
        audio_embs = audio_embs + 1e-8

        # Apply transformer with stability check
        encoded = self.audio_transformer(audio_embs)

        # Simple mean pooling is more stable
        pooled = encoded.mean(dim=1)

        # Check for NaNs and replace if needed
        if torch.isnan(pooled).any():
            pooled = torch.where(torch.isnan(pooled), torch.ones_like(pooled) * 1e-8, pooled)

        projected = self.audio_projection(pooled)

        # Softly normalize to avoid division by zero
        norm = torch.norm(projected, p=2, dim=1, keepdim=True).clamp(min=1e-8)
        normalized = projected / norm

        return normalized

    def encode_text(self, text_embs):
        # Handle different input dimensions
        if text_embs.dim() == 1:
            text_embs = text_embs.unsqueeze(0)
        if text_embs.dim() == 2:
            text_embs = text_embs.unsqueeze(1)

        # Add small epsilon to prevent zeros
        text_embs = text_embs + 1e-8

        # Apply transformer with stability check
        encoded = self.text_transformer(text_embs)

        # Simple mean pooling is more stable
        pooled = encoded.mean(dim=1)

        # Check for NaNs and replace if needed
        if torch.isnan(pooled).any():
            pooled = torch.where(torch.isnan(pooled), torch.ones_like(pooled) * 1e-8, pooled)

        projected = self.text_projection(pooled)

        # Softly normalize to avoid division by zero
        norm = torch.norm(projected, p=2, dim=1, keepdim=True).clamp(min=1e-8)
        normalized = projected / norm

        return normalized

    def forward(self, audio_embs, text_embs):
        audio_encoding = self.encode_speech(audio_embs)
        text_encoding = self.encode_text(text_embs)
        return audio_encoding, text_encoding

    @staticmethod
    def correlation_loss(representations: list, eps=EPS):
        """
        representations: [z1, z2] from two encoders. Both tensors are [batch_size, latent_dim]
        Returns: negative mean of canonical correlations
        """
        z1, z2 = representations
        z1 = z1 - z1.mean(dim=0, keepdim=True)
        z2 = z2 - z2.mean(dim=0, keepdim=True)

        cov_z1 = z1.T @ z1 / (z1.size(0) - 1) + eps * torch.eye(z1.size(1), device=z1.device)
        cov_z2 = z2.T @ z2 / (z2.size(0) - 1) + eps * torch.eye(z2.size(1), device=z2.device)
        cross_cov = z1.T @ z2 / (z1.size(0) - 1)

        sym_cross_cov = (cross_cov + cross_cov.T) / 2
        eigvals = torch.linalg.eigh(torch.linalg.solve(cov_z1, sym_cross_cov) @ torch.linalg.solve(cov_z2, sym_cross_cov.T))[0]
        canonical_corrs = torch.sqrt(torch.clamp(eigvals, min=0.0))
        return -canonical_corrs.mean()

    @staticmethod
    def loss(audio_encodings, text_encodings):
        return DCCAV2SpeechText.correlation_loss([audio_encodings, text_encodings])

    @staticmethod
    def contrastive_loss(audio_encodings, text_encodings, neg_text_encodings):
        pos_correlation_loss = DCCAV2SpeechText.correlation_loss([audio_encodings, text_encodings])
        neg_correlation_loss = DCCAV2SpeechText.correlation_loss([audio_encodings, neg_text_encodings])
        return pos_correlation_loss - neg_correlation_loss


def create_dcca_v2_model(audio_embedding_dim=AUDIO_EMBED_DIM, text_embedding_dim=TEXT_EMBED_DIM, shared_dim=SHARED_DIM,
                      device=DEVICE, state_dict=None):
    model = DCCAV2SpeechText(audio_dim=audio_embedding_dim, text_dim=text_embedding_dim, shared_dim=shared_dim, device=device)
    model.to(device)
    if state_dict is not None:
        model.load_state_dict(state_dict)
        print("✅ Loaded model state dict.")
    else:
        print("ℹ️ No state dict provided. Initializing model with random weights.")
    return model


# ========== Training ==========

def collate_fn(batch):
    audio_embeddings = torch.tensor([item[RESEMBLYZER_SPEAKER_EMBEDDING_COLUMN] for item in batch])
    description_embeddings = torch.tensor([item[GRANITE_DESCRIPTION_EMBEDDING_COLUMN] for item in batch])
    neg_description_embeddings = torch.tensor([item[GRANITE_NEGATIVE_DESCRIPTION_EMBEDDING_COLUMN] for item in batch])

    for name, tensor in zip(["audio", "desc", "neg_desc"], [audio_embeddings, description_embeddings, neg_description_embeddings]):
        if not torch.isfinite(tensor).all():
            print(f"⚠️ Non-finite or NaN values detected in {name} embeddings. Skipping batch.")
            return None

    return audio_embeddings.squeeze(1).to(DEVICE), description_embeddings.squeeze(1).to(DEVICE), neg_description_embeddings.squeeze(1).to(DEVICE)


def evaluate(model, dataloader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluation"):
            if batch is None:
                continue
            audio_emb, desc_emb, neg_desc_emb = batch
            audio_encoding = model.encode_speech(audio_emb)
            desc_encoding = model.encode_text(desc_emb)
            neg_desc_encoding = model.encode_text(neg_desc_emb)
            loss = model.contrastive_loss(audio_encoding, desc_encoding, neg_desc_encoding)
            total_loss += loss.item()
    return total_loss / len(dataloader)


def train():
    dataset_dict = load_from_disk(ENRICHED_DATASET_V2_PATH)
    train_dataset = dataset_dict["train"].cast_column(AUDIO_COLUMN, Audio())
    test_dataset = dataset_dict["test"].cast_column(AUDIO_COLUMN, Audio())

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    model = create_dcca_v2_model()

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.3, patience=5)

    best_val_loss = float("inf")
    train_losses, val_losses = [], []

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_train_loss = 0
        for batch in tqdm(train_loader, desc=f" Training Epoch {epoch + 1}/{NUM_EPOCHS}"):
            if batch is None:
                continue
            audio_emb, desc_emb, neg_desc_emb = batch
            audio_encoding = model.encode_speech(audio_emb)
            desc_encoding = model.encode_text(desc_emb)
            neg_desc_encoding = model.encode_text(neg_desc_emb)
            loss = model.contrastive_loss(audio_encoding, desc_encoding, neg_desc_encoding)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = evaluate(model, val_loader)
        scheduler.step(avg_val_loss)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch + 1} - Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "val_loss": best_val_loss}, DCCA_V2_MODEL_PATH)
            print("✅ Saved new best model.")

    print("🏁 Training complete.")
    print("🔍 Evaluating best model on validation set...")
    model.load_state_dict(torch.load(DCCA_V2_MODEL_PATH)["model_state_dict"])
    final_val_loss = evaluate(model, val_loader)
    print(f"Best Model Validation Loss: {final_val_loss:.4f}")

    model_data = torch.load(DCCA_V2_MODEL_PATH)
    model_data["training_losses"] = train_losses
    model_data["validation_losses"] = val_losses
    torch.save(model_data, DCCA_V2_MODEL_PATH)


if __name__ == "__main__":
    train()

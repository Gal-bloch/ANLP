import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_from_disk, Audio
from tqdm import tqdm
from create_dataset import ENRICHED_DATASET_V2_PATH, AUDIO_COLUMN, RESEMBLYZER_SPEAKER_EMBEDDING_COLUMN, GRANITE_DESCRIPTION_EMBEDDING_COLUMN, GRANITE_NEGATIVE_DESCRIPTION_EMBEDDING_COLUMN
from resemblyzer import VoiceEncoder
from sentence_transformers import SentenceTransformer


# ========== CONFIG ==========
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
DCCA_V3_VOICE_EMBEDDER = VoiceEncoder(device=DEVICE)
DCCA_V3_DESCRIPTION_EMBEDDER = SentenceTransformer("ibm-granite/granite-embedding-125m-english")


BATCH_SIZE = 32  # Increased batch size for better contrastive learning
NUM_EPOCHS = 50
LR = 1e-4  # Adjusted learning rate
WEIGHT_DECAY = 1e-5
DROPOUT_RATE = 0.15  # Slightly increased dropout
HEAD_NUM = 8
LAYER_NUM = 5  # Deeper transformer
EPS = 1e-6
TEXT_EMBED_DIM = 768
AUDIO_EMBED_DIM = 256
SHARED_DIM = 256
TEMPERATURE = 0.07  # For InfoNCE loss
DCCA_V3_MODEL_PATH = "../models/dccav3_voice_text.pt"


# ========== Model Definition ==========

class DCCAV3SpeechText(nn.Module):
    def __init__(self, audio_dim=AUDIO_EMBED_DIM, text_dim=TEXT_EMBED_DIM, shared_dim=SHARED_DIM,
                 dropout_rate=DROPOUT_RATE, nhead=HEAD_NUM, num_layers=LAYER_NUM,
                 device=DEVICE):
        super(DCCAV3SpeechText, self).__init__()
        self.device = device
        self.shared_dim = shared_dim

        # Audio Tower
        self.audio_norm = nn.LayerNorm(audio_dim)
        audio_encoder_layer = nn.TransformerEncoderLayer(
            d_model=audio_dim,
            nhead=nhead,
            dim_feedforward=4 * audio_dim,
            dropout=dropout_rate,
            batch_first=True
        )
        self.audio_transformer = nn.TransformerEncoder(audio_encoder_layer, num_layers=num_layers)
        self.audio_projection = nn.Sequential(
            nn.LayerNorm(audio_dim),
            nn.Linear(audio_dim, 2 * shared_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(2 * shared_dim, shared_dim)
        )

        # Text Tower
        self.text_norm = nn.LayerNorm(text_dim)
        text_encoder_layer = nn.TransformerEncoderLayer(
            d_model=text_dim,
            nhead=nhead,
            dim_feedforward=4 * text_dim,
            dropout=dropout_rate,
            batch_first=True
        )
        self.text_transformer = nn.TransformerEncoder(text_encoder_layer, num_layers=num_layers)
        self.text_projection = nn.Sequential(
            nn.LayerNorm(text_dim),
            nn.Linear(text_dim, 2 * shared_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(2 * shared_dim, shared_dim)
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def encode_speech(self, audio_embs):
        if audio_embs.dim() == 2:
            audio_embs = audio_embs.unsqueeze(1)

        audio_embs = self.audio_norm(audio_embs)
        encoded = self.audio_transformer(audio_embs)
        pooled = encoded.mean(dim=1)
        projected = self.audio_projection(pooled)
        normalized = nn.functional.normalize(projected, p=2, dim=1)
        return normalized

    def encode_text(self, text_embs):
        if text_embs.dim() == 2:
            text_embs = text_embs.unsqueeze(1)

        text_embs = self.text_norm(text_embs)
        encoded = self.text_transformer(text_embs)
        pooled = encoded.mean(dim=1)
        projected = self.text_projection(pooled)
        normalized = nn.functional.normalize(projected, p=2, dim=1)
        return normalized

    def forward(self, audio_embs, text_embs):
        audio_encoding = self.encode_speech(audio_embs)
        text_encoding = self.encode_text(text_embs)
        return audio_encoding, text_encoding

    @staticmethod
    def info_nce_loss(features1, features2, temperature=TEMPERATURE):
        batch_size = features1.shape[0]
        device = features1.device

        # Calculate cosine similarity
        logits = (features1 @ features2.T) / temperature
        
        # Create labels for positive pairs (diagonal elements)
        labels = torch.arange(batch_size, device=device)
        
        # Loss is cross-entropy between predicted similarities and true pairs
        loss = nn.CrossEntropyLoss()(logits, labels)
        return loss

    @staticmethod
    def loss(audio_encodings, text_encodings, neg_text_encodings, temperature=TEMPERATURE):
        # Symmetric InfoNCE loss
        loss_audio_text = DCCAV3SpeechText.info_nce_loss(audio_encodings, text_encodings, temperature)
        loss_text_audio = DCCAV3SpeechText.info_nce_loss(text_encodings, audio_encodings, temperature)
        
        # Combine positive pair losses
        positive_loss = (loss_audio_text + loss_text_audio) / 2

        return positive_loss


def create_dcca_v3_model(audio_embedding_dim=AUDIO_EMBED_DIM, text_embedding_dim=TEXT_EMBED_DIM, shared_dim=SHARED_DIM,
                      device=DEVICE, state_dict=None):
    model = DCCAV3SpeechText(audio_dim=audio_embedding_dim, text_dim=text_embedding_dim, shared_dim=shared_dim, device=device)
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

    return audio_embeddings.squeeze(1).to(DEVICE), description_embeddings.squeeze(1).to(DEVICE), neg_description_embeddings.squeeze(1).to(DEVICE)


def evaluate(model, dataloader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluation"):
            audio_emb, desc_emb, neg_desc_emb = batch
            audio_encoding = model.encode_speech(audio_emb)
            desc_encoding = model.encode_text(desc_emb)
            neg_desc_encoding = model.encode_text(neg_desc_emb)
            loss = model.loss(audio_encoding, desc_encoding, neg_desc_encoding)
            total_loss += loss.item()
    return total_loss / len(dataloader)


def train():
    dataset_dict = load_from_disk(ENRICHED_DATASET_V2_PATH)
    train_dataset = dataset_dict["train"].cast_column(AUDIO_COLUMN, Audio())
    test_dataset = dataset_dict["test"].cast_column(AUDIO_COLUMN, Audio())

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    model = create_dcca_v3_model()

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS * len(train_loader))

    best_val_loss = float("inf")
    train_losses, val_losses = [], []

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_train_loss = 0
        for batch in tqdm(train_loader, desc=f" Training Epoch {epoch + 1}/{NUM_EPOCHS}"):
            audio_emb, desc_emb, neg_desc_emb = batch
            
            optimizer.zero_grad()
            
            audio_encoding = model.encode_speech(audio_emb)
            desc_encoding = model.encode_text(desc_emb)
            neg_desc_encoding = model.encode_text(neg_desc_emb)
            
            loss = model.loss(audio_encoding, desc_encoding, neg_desc_encoding)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = evaluate(model, val_loader)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch + 1} - Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "val_loss": best_val_loss}, DCCA_V3_MODEL_PATH)
            print("✅ Saved new best model.")

    print("🏁 Training complete.")
    print("🔍 Evaluating best model on validation set...")
    model.load_state_dict(torch.load(DCCA_V3_MODEL_PATH)["model_state_dict"])
    final_val_loss = evaluate(model, val_loader)
    print(f"Best Model Validation Loss: {final_val_loss:.4f}")

    model_data = torch.load(DCCA_V3_MODEL_PATH)
    model_data["training_losses"] = train_losses
    model_data["validation_losses"] = val_losses
    torch.save(model_data, DCCA_V3_MODEL_PATH)


if __name__ == "__main__":
    train()

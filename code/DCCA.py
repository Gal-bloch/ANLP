import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.tensorboard.summary
from torch.utils.data import DataLoader
from datasets import load_from_disk, Audio
from tqdm import tqdm
from cca_zoo.deep import DCCA, DCCA_EY
from cca_zoo.deep import architectures
from create_dataset import ENRICHED_DATASET_V2_PATH, AUDIO_COLUMN, RESEMBLYZER_SPEAKER_EMBEDDING_COLUMN, GRANITE_DESCRIPTION_EMBEDDING_COLUMN, GRANITE_NEGATIVE_DESCRIPTION_EMBEDDING_COLUMN
from Loss_Functions import contrastive_loss
from resemblyzer import VoiceEncoder
from sentence_transformers import SentenceTransformer


# ========== CONFIG ==========
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
DCCA_VOICE_EMBEDDER = VoiceEncoder(device=DEVICE)
DCCA_DESCRIPTION_EMBEDDER = SentenceTransformer("ibm-granite/granite-embedding-125m-english")


BATCH_SIZE = 16
NUM_EPOCHS = 50
LR = 1e-3
WEIGHT_DECAY = 1e-4
TEXT_EMBED_DIM = 768
AUDIO_EMBED_DIM = 256  # VoiceEncoder outputs 256-dim embeddings
SHARED_DIM = 128  # Dimension of the shared space
AUDIO_PATH = "../datasets/dataset_with_negations"
DCCA_MODEL_PATH = "../models/dcca_voice_text.pt"


# ========== Model Definition ==========

def create_dcca_model(audio_embedding_dim=AUDIO_EMBED_DIM, text_embedding_dim=TEXT_EMBED_DIM, shared_dim=SHARED_DIM,
                      device=DEVICE, state_dict=None):

    audio_encoder = architectures.Encoder(latent_dimensions=shared_dim, feature_size=audio_embedding_dim,
                                          layer_sizes=(8 * audio_embedding_dim, 4 * audio_embedding_dim, 2 * audio_embedding_dim),
                                            activation=nn.ReLU(), dropout=0.1)
    text_encoder = architectures.Encoder(latent_dimensions=shared_dim, feature_size=text_embedding_dim,
                                        layer_sizes=(8 * text_embedding_dim, 4 * text_embedding_dim, 2 * text_embedding_dim),
                                            activation=nn.ReLU(), dropout=0.1)

    model = DCCA(
        latent_dimensions=shared_dim,
        encoders=[audio_encoder, text_encoder],
        device=device
    ).to(device)

    if state_dict is not None:
        try:
            model.load_state_dict(state_dict)
            print("✅ Successfully loaded model weights")
        except Exception as e:
            print(f"⚠️ Error loading model weights: {e}")

    return model

# ========== Training ==========

def collate_fn(batch):
    """
    Custom collate function to handle variable-length audio and text embeddings.
    """
    audio_embeddings = torch.tensor([item[RESEMBLYZER_SPEAKER_EMBEDDING_COLUMN] for item in batch])
    description_embeddings = torch.tensor([item[GRANITE_DESCRIPTION_EMBEDDING_COLUMN] for item in batch])
    neg_description_embeddings = torch.tensor([item[GRANITE_NEGATIVE_DESCRIPTION_EMBEDDING_COLUMN] for item in batch])

    return audio_embeddings, description_embeddings, neg_description_embeddings


def evaluate(model, dataloader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for audio_emb, desc_emb, neg_desc_emb in dataloader:
            audio_emb = audio_emb.to(DEVICE)
            desc_emb = desc_emb.to(DEVICE)
            neg_desc_emb = neg_desc_emb.to(DEVICE)

            loss_list = model.forward([audio_emb, desc_emb])
            loss = loss_list[0].mean()

            audio_latent = model.encoders[0](audio_emb)
            neg_latent = model.encoders[1](neg_desc_emb)
            neg_sim = F.cosine_similarity(audio_latent, neg_latent)
            contrastive_loss = F.relu(neg_sim - 0.5).mean()
            loss += contrastive_loss

            total_loss += loss.item()

    return total_loss / len(dataloader)


def train():
    dataset = load_from_disk(ENRICHED_DATASET_V2_PATH).cast_column(AUDIO_COLUMN, Audio())
    dataset = dataset.train_test_split(test_size=0.1, seed=42)

    train_loader = DataLoader(dataset["train"], batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(dataset["test"], batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    model = create_dcca_model()


    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.3, patience=5)

    best_val_loss = float("inf")

    train_losses, val_losses = [], []
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_train_loss = 0
        for audio_emb, desc_emb, neg_desc_emb in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}"):
            audio_emb = audio_emb.to(DEVICE)
            desc_emb = desc_emb.to(DEVICE)
            neg_desc_emb = neg_desc_emb.to(DEVICE)

            loss_list = model.forward([audio_emb, desc_emb])
            loss = loss_list[0].mean()

            audio_latent = model.encoders[0](audio_emb)
            desc_latent = model.encoders[1](desc_emb)
            neg_desc_latent = model.encoders[1](neg_desc_emb)
            contrast_loss = contrastive_loss(audio_latent, desc_latent, neg_desc_latent)

            loss += contrast_loss


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = evaluate(model, val_loader)
        scheduler.step(avg_val_loss)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch + 1} - Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # Save the best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save( {"epoch": epoch, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
             "val_loss": best_val_loss}, DCCA_MODEL_PATH)
            print("✅ Saved new best model.")



    print("🏁 Training complete.")

    # ===== TESTING =====
    print("🔍 Evaluating best model on validation set...")
    model.load_state_dict(torch.load(DCCA_MODEL_PATH)["model_state_dict"])
    final_val_loss = evaluate(model, val_loader)
    print(f"Best Model Validation Loss: {final_val_loss:.4f}")

    model_data = torch.load(DCCA_MODEL_PATH)
    model_data["training_losses"] = train_losses
    model_data["validation_losses"] = val_losses
    torch.save(model_data, DCCA_MODEL_PATH)


if __name__ == "__main__":
    train()
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.tensorboard.summary
from torch.utils.data import DataLoader
from datasets import load_from_disk, Audio
from tqdm import tqdm
from cca_zoo.deep import DCCAE
from cca_zoo.deep import architectures
from create_dataset import ENRICHED_DATASET_PATH, AUDIO_COLUMN, SPEAKER_EMBEDDING_COLUMN, DESCRIPTION_EMBEDDING_COLUMN, NEGATIVE_DESCRIPTION_EMBEDDING_COLUMN


# ========== CONFIG ==========
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
BATCH_SIZE = 16
NUM_EPOCHS = 50
LR = 1e-2
TEXT_EMBED_DIM = 768
AUDIO_EMBED_DIM = 256  # VoiceEncoder outputs 256-dim embeddings
SHARED_DIM = 128  # Dimension of the shared space
AUDIO_PATH = "/Users/galbloch/Desktop/school/git/ANLP/datasets/dataset_with_negations"


# ========== DATASET ==========

def collate_fn(batch):
    """
    Custom collate function to handle variable-length audio and text embeddings.
    """
    audio_embeddings = torch.tensor([item[SPEAKER_EMBEDDING_COLUMN] for item in batch])
    description_embeddings = torch.tensor([item[DESCRIPTION_EMBEDDING_COLUMN] for item in batch])
    neg_description_embeddings = torch.tensor([item[NEGATIVE_DESCRIPTION_EMBEDDING_COLUMN] for item in batch])



    return audio_embeddings, description_embeddings, neg_description_embeddings


# ========== TRAIN ==========

def create_dccae_model(audio_embedding_dim=AUDIO_EMBED_DIM, text_embedding_dim=TEXT_EMBED_DIM, shared_dim=SHARED_DIM,
                          device=DEVICE, state_dict=None):
    audio_encoder = architectures.Encoder(latent_dimensions=shared_dim, feature_size=audio_embedding_dim)
    text_encoder = architectures.Encoder(latent_dimensions=shared_dim, feature_size=text_embedding_dim)
    audio_decoder = architectures.Decoder(latent_dimensions=shared_dim, feature_size=audio_embedding_dim)
    text_decoder = architectures.Decoder(latent_dimensions=shared_dim, feature_size=text_embedding_dim)

    model = DCCAE(
        latent_dimensions=shared_dim,
        encoders=[audio_encoder, text_encoder],
        decoders=[audio_decoder, text_decoder],
        device=device
    ).to(device)

    if state_dict is not None:
        try:
            model.load_state_dict(state_dict)
            print("✅ Successfully loaded model weights")
        except Exception as e:
            print(f"⚠️ Error loading model weights: {e}")

    return model


def train():
    dataset = load_from_disk(ENRICHED_DATASET_PATH).cast_column(AUDIO_COLUMN, Audio())
    dataset = dataset.train_test_split(test_size=0.1, seed=42)

    train_loader = DataLoader(dataset["train"], batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(dataset["test"], batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    model = create_dccae_model()


    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_loss = float("inf")
    best_model_path = "../models/dccae_voice_text_best.pt"

    def evaluate(dataloader, use_negatives=False):
        model.eval()
        total_loss = 0
        with torch.no_grad():
            for audio_emb, desc_emb, neg_desc_emb in dataloader:
                audio_emb = audio_emb.to(DEVICE)
                desc_emb = desc_emb.to(DEVICE)
                neg_desc_emb = neg_desc_emb.to(DEVICE)

                loss_list = model.forward([audio_emb, desc_emb])
                loss = loss_list[0].mean()

                if use_negatives:
                    audio_latent = model.encoders[0](audio_emb)
                    neg_latent = model.encoders[1](neg_desc_emb)
                    neg_sim = F.cosine_similarity(audio_latent, neg_latent)
                    contrastive_loss = F.relu(neg_sim - 0.5).mean()
                    loss += contrastive_loss



                total_loss += loss.item()
        return total_loss / len(dataloader)

    # ===== TRAINING LOOP =====
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
            neg_latent = model.encoders[1](neg_desc_emb)
            neg_sim = F.cosine_similarity(audio_latent, neg_latent)
            contrastive_loss = F.relu(neg_sim - 0.5).mean()

            loss += contrastive_loss


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = evaluate(val_loader, use_negatives=True)

        print(f"Epoch {epoch + 1} - Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # Save the best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save( {"epoch": epoch, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
             "val_loss": best_val_loss}, best_model_path)
            print("✅ Saved new best model.")



    print("🏁 Training complete.")

    # ===== TESTING =====
    print("🔍 Evaluating best model on validation set...")
    model.load_state_dict(torch.load(best_model_path)["model_state_dict"])
    final_val_loss = evaluate(val_loader, use_negatives=True)
    print(f"Best Model Validation Loss: {final_val_loss:.4f}")


if __name__ == "__main__":
    train()
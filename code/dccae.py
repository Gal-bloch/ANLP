import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_from_disk, Audio
from sentence_transformers import SentenceTransformer
from resemblyzer import VoiceEncoder, wav_to_mel_spectrogram
from tqdm import tqdm
import numpy as np
from cca_zoo.deep import DCCAE
from cca_zoo.deep import architectures





# ========== CONFIG ==========
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16
NUM_EPOCHS = 50
LR = 1e-3
TEXT_EMBED_DIM = 768
AUDIO_EMBED_DIM = 256  # VoiceEncoder outputs 256-dim embeddings
SHARED_DIM = 128  # Dimension of the shared space
AUDIO_PATH = "/Users/galbloch/Desktop/school/git/ANLP/datasets/dataset_with_negations"


# ========== DATASET ==========

def get_text_embedding(texts, model):
    with torch.no_grad():
        embeddings = model.encode(texts, convert_to_tensor=True)
        return F.normalize(embeddings, p=2, dim=1)


def collate_fn(batch):
    """
    Collate function for DataLoader that processes audio data into mel spectrograms.
    """
    wavs = [item["audio"]["array"] for item in batch]
    descriptions = [item["text_description"] for item in batch]
    neg_descriptions = [item["negated_prompt"] for item in batch]

    # Process each audio waveform into a mel spectrogram
    mel_specs = []
    for wav in wavs:
        try:
            mel = wav_to_mel_spectrogram(wav)
            mel_specs.append(mel)
        except Exception as e:
            print(f"Error processing audio to mel: {e}")
            mel_specs.append(np.zeros((80, 40)))

    return mel_specs, descriptions, neg_descriptions


# ========== MODEL ==========

class AudioTextDCCAE(nn.Module):
    def __init__(self, audio_dim=AUDIO_EMBED_DIM, text_dim=TEXT_EMBED_DIM, shared_dim=SHARED_DIM, hidden_dim=512):
        super().__init__()

        # Encoders that map to shared space
        self.audio_encoder = nn.Sequential(
            nn.Linear(audio_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, shared_dim)
        )

        self.text_encoder = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, shared_dim)
        )

        # Decoders for reconstruction
        self.audio_decoder = nn.Sequential(
            nn.Linear(shared_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, audio_dim)
        )

        self.text_decoder = nn.Sequential(
            nn.Linear(shared_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, text_dim)
        )

    def forward(self, audio_emb, text_emb):
        shared_audio = self.audio_encoder(audio_emb)
        shared_text = self.text_encoder(text_emb)

        audio_recon = self.audio_decoder(shared_audio)
        text_recon = self.text_decoder(shared_text)

        return shared_audio, shared_text, audio_recon, text_recon


# ========== TRAIN ==========

def train():
    dataset = load_from_disk(AUDIO_PATH).cast_column("audio", Audio())

    if "negated_prompt" not in dataset.column_names:
        raise ValueError("Dataset must contain 'negated_prompt' field")

    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    train_loader = DataLoader(dataset["train"], batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(dataset["test"], batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    granite = SentenceTransformer("ibm-granite/granite-embedding-125m-english")
    voice_encoder = VoiceEncoder(device=DEVICE)

    def encode_audio_batch(mels):
        embeddings = []
        for mel in mels:
            try:
                mel_np = mel.detach().cpu().numpy() if isinstance(mel, torch.Tensor) else mel
                if len(mel_np.shape) == 2:
                    emb = voice_encoder(torch.tensor(mel_np).unsqueeze(0).to(DEVICE)).cpu().numpy()[0]
                else:
                    reshaped = mel_np.reshape(-1, mel_np.shape[-1]) if mel_np.size > 0 else mel_np
                    emb = voice_encoder(torch.tensor(reshaped).unsqueeze(0).to(DEVICE)).cpu().numpy()[0]
                embeddings.append(torch.tensor(emb))
            except Exception as e:
                print(f"Error embedding mel spectrogram: {e}")
                embeddings.append(torch.zeros(AUDIO_EMBED_DIM))
        return torch.stack(embeddings).to(DEVICE)

    use_cca_zoo = True

    if use_cca_zoo:
        audio_encoder = architectures.Encoder(latent_dimensions=SHARED_DIM, feature_size=AUDIO_EMBED_DIM)
        text_encoder = architectures.Encoder(latent_dimensions=SHARED_DIM, feature_size=TEXT_EMBED_DIM)
        audio_decoder = architectures.Decoder(latent_dimensions=SHARED_DIM, feature_size=AUDIO_EMBED_DIM)
        text_decoder = architectures.Decoder(latent_dimensions=SHARED_DIM, feature_size=TEXT_EMBED_DIM)

        model = DCCAE(
            latent_dimensions=SHARED_DIM,
            encoders=[audio_encoder, text_encoder],
            decoders=[audio_decoder, text_decoder],
            device=DEVICE
        ).to(DEVICE)
    else:
        model = AudioTextDCCAE().to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_loss = float("inf")
    best_model_path = "dccae_voice_text_best.pt"

    def evaluate(dataloader, use_negatives=False):
        model.eval()
        total_loss = 0
        with torch.no_grad():
            for mel_specs, pos_texts, neg_texts in dataloader:
                audio_emb = encode_audio_batch(mel_specs)
                text_emb = get_text_embedding(pos_texts, granite).to(DEVICE)
                neg_emb = get_text_embedding(neg_texts, granite).to(DEVICE)

                if use_cca_zoo:
                    loss_list = model.forward([audio_emb, text_emb])
                    loss = loss_list[0].mean()

                    if use_negatives:
                        audio_latent = model.encoders[0](audio_emb)
                        neg_latent = model.encoders[1](neg_emb)
                        neg_sim = F.cosine_similarity(audio_latent, neg_latent)
                        contrastive_loss = F.relu(neg_sim - 0.5).mean()
                        loss += contrastive_loss
                else:
                    shared_audio, shared_text, audio_recon, text_recon = model(audio_emb, text_emb)
                    recon_loss = F.mse_loss(audio_recon, audio_emb) + F.mse_loss(text_recon, text_emb)
                    pos_sim = F.cosine_similarity(shared_audio, shared_text)
                    neg_sim = F.cosine_similarity(shared_audio, model.text_encoder(neg_emb))
                    contrastive = (1 - pos_sim).mean() + F.relu(neg_sim - 0.5).mean()
                    loss = recon_loss + contrastive

                total_loss += loss.item()
        return total_loss / len(dataloader)

    # ===== TRAINING LOOP =====
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_train_loss = 0
        for mel_specs, pos_texts, neg_texts in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}"):
            with torch.no_grad():
                audio_emb = encode_audio_batch(mel_specs)
                text_emb = get_text_embedding(pos_texts, granite).to(DEVICE)
                neg_emb = get_text_embedding(neg_texts, granite).to(DEVICE)

            if use_cca_zoo:
                loss_list = model.forward([audio_emb, text_emb])
                loss = loss_list[0].mean()

                audio_latent = model.encoders[0](audio_emb)
                neg_latent = model.encoders[1](neg_emb)
                neg_sim = F.cosine_similarity(audio_latent, neg_latent)
                contrastive_loss = F.relu(neg_sim - 0.5).mean()

                loss += contrastive_loss
            else:
                shared_audio, shared_text, audio_recon, text_recon = model(audio_emb, text_emb)
                recon_loss = F.mse_loss(audio_recon, audio_emb) + F.mse_loss(text_recon, text_emb)
                pos_sim = F.cosine_similarity(shared_audio, shared_text)
                neg_sim = F.cosine_similarity(shared_audio, model.text_encoder(neg_emb))
                contrastive = (1 - pos_sim).mean() + F.relu(neg_sim - 0.5).mean()
                loss = recon_loss + contrastive

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
            torch.save(model.state_dict(), best_model_path)
            print("✅ Saved new best model.")

    print("🏁 Training complete.")

    # ===== TESTING =====
    print("🔍 Evaluating best model on validation set...")
    model.load_state_dict(torch.load(best_model_path))
    final_val_loss = evaluate(val_loader, use_negatives=True)
    print(f"Best Model Validation Loss: {final_val_loss:.4f}")


if __name__ == "__main__":
    train()
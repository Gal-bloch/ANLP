import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm
from datasets import load_from_disk, Audio, concatenate_datasets

from create_dataset import ENRICHED_DATASET_V2_PATH, AUDIO_COLUMN, GRANITE_DESCRIPTION_EMBEDDING_COLUMN, RESEMBLYZER_SPEAKER_EMBEDDING_COLUMN
from create_dataset import GENDER_COLUMN, AGE_COLUMN, PITCH_COLUMN, EMOTION_COLUMN, SPEED_COLUMN, ENERGY_COLUMN
import os
from DCCA import create_dcca_model, DCCA_MODEL_PATH
from DCCAV2 import create_dcca_v2_model, DCCA_V2_MODEL_PATH
from DCCAV3 import create_dcca_v3_model, DCCA_V3_MODEL_PATH
from Voice2Embedding import Voice2Embedding, VOICE2EMBEDDING_MODEL_PATH
import pandas as pd


# Constants
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_TYPES = ["dccav3", "dccav2", "dcca", "voice2embedding"]

def get_ordered_matches(text_embedding, speech_embeddings):
    text_embedding = torch.tensor(text_embedding, dtype=torch.float32).to(DEVICE).squeeze()
    speech_embeddings = torch.tensor(speech_embeddings, dtype=torch.float32).to(DEVICE)

    if text_embedding.dim() != 1 and speech_embeddings.dim() != 2:
        raise ValueError("text_embedding must be 1D and speech_embeddings must be 2D")

    text_embeddings = text_embedding.unsqueeze(0).repeat(speech_embeddings.shape[0], 1)
    if text_embeddings.shape[1] != speech_embeddings.shape[1]:
        raise ValueError("Text and speech embeddings must have the same dimension")

    sim = F.cosine_similarity(text_embeddings, speech_embeddings, dim=1)
    ordered_indices = torch.argsort(sim, descending=True)

    return ordered_indices

def load_model(model_type):
    if model_type == "dccav3":
        model_state_dict = torch.load(DCCA_V3_MODEL_PATH, map_location=DEVICE)["model_state_dict"]
        model = create_dcca_v3_model(state_dict=model_state_dict)
        model.eval()
        return model
    elif model_type == "dccav2":
        model_state_dict = torch.load(DCCA_V2_MODEL_PATH, map_location=DEVICE)["model_state_dict"]
        model = create_dcca_v2_model(state_dict=model_state_dict)
        model.eval()
        return model
    elif model_type == "dcca":
        model_state_dict = torch.load(DCCA_MODEL_PATH, map_location=DEVICE)["model_state_dict"]
        model = create_dcca_model(state_dict=model_state_dict)
        model.eval()
        return model
    elif model_type == "voice2embedding":
        model_state_dict = torch.load(VOICE2EMBEDDING_MODEL_PATH, map_location=DEVICE)["model_state_dict"]
        model = Voice2Embedding()
        model.load_state_dict(model_state_dict)
        model.eval()
        return model



def get_text_encodings(dataset, model, model_type):
    if model_type == "voice2embedding":
        return torch.tensor(dataset[GRANITE_DESCRIPTION_EMBEDDING_COLUMN], dtype=torch.float32).to(DEVICE)
    else:
        text_embeddings = torch.tensor(dataset[GRANITE_DESCRIPTION_EMBEDDING_COLUMN], dtype=torch.float32).to(DEVICE)
        return model.encode_text(text_embeddings)

def get_speech_encodings(dataset, model):
    speech_embeddings = torch.tensor(dataset[RESEMBLYZER_SPEAKER_EMBEDDING_COLUMN], dtype=torch.float32).to(DEVICE)
    return model.encode_speech(speech_embeddings)



def evaluate_model_type(model_type):
    if model_type not in MODEL_TYPES:
        raise ValueError(f"Invalid model type. Choose from {MODEL_TYPES}")

    model = load_model(model_type)
    dataset = load_from_disk(ENRICHED_DATASET_V2_PATH)["test"].cast_column(AUDIO_COLUMN, Audio())

    text_encodings = get_text_encodings(dataset, model, model_type)
    speech_encodings = get_speech_encodings(dataset, model)

    score = 0
    for i, text_encoding in tqdm(enumerate(text_encodings), total=len(text_encodings), desc=f"Evaluating {model_type}"):
        ordered_indices = get_ordered_matches(text_encoding, speech_encodings)

        # find is index in ordered_indices
        current_score = ((len(speech_encodings) - 1) - ((ordered_indices == i).nonzero(as_tuple=True)[0].item())) / (len(speech_encodings) - 1)
        score += current_score

    return score / len(text_encodings)

if __name__ == "__main__":
    results = {}
    for model_type in MODEL_TYPES:
        score = evaluate_model_type(model_type)
        results[model_type] = score
        print(f"{model_type} score: {score:.4f}")

    # Save results to CSV
    results_df = pd.DataFrame.from_dict(results, orient='index', columns=['Score'])
    results_df.to_csv("../results/retrieval_evaluation_results.csv", index_label='Model Type')
    print("Results saved to ../results/retrieval_evaluation_results.csv")








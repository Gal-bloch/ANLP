import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
from tqdm import tqdm
from datasets import load_from_disk, Audio
from create_dataset import ENRICHED_DATASET_PATH, SPEAKER_EMBEDDING_COLUMN, DESCRIPTION_EMBEDDING_COLUMN, NEGATIVE_DESCRIPTION_EMBEDDING_COLUMN, GENDER_COLUMN, PITCH_COLUMN
import os
from dccae import create_dccae_model

# Constants
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 1e-3
HIDDEN_DIM = 128
EARLY_STOPPING_PATIENCE = 5

class FeatureClassificationDataset(Dataset):
    """Dataset wrapper for gender classification task."""
    def __init__(self, embeddings, labels):
        self.embeddings = torch.tensor(embeddings).to(DEVICE)
        self.labels = torch.tensor(labels).long().to(DEVICE)
    
    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]

class SpeechFeatureClassifier(nn.Module):
    """
    A simple binary classifier for single modality data.
    """
    def __init__(self, input_dim =128, output_dim=2):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
        )

    def forward(self, x):
        return self.classifier(x)

    def predict(self, x):
        """
        Predict the class probabilities for the input data.
        Args:
            x: Input data tensor.
        Returns:
            Tensor of predicted classes assigned to the input data.
        """
        with torch.no_grad():
            return torch.argmax(self.forward(x), dim=1)

def collate_fn(batch):
    """
    Custom collate function to handle variable-length audio and text embeddings.
    """
    audio_embeddings = torch.tensor([item[SPEAKER_EMBEDDING_COLUMN] for item in batch])
    description_embeddings = torch.tensor([item[DESCRIPTION_EMBEDDING_COLUMN] for item in batch])
    neg_description_embeddings = torch.tensor([item[NEGATIVE_DESCRIPTION_EMBEDDING_COLUMN] for item in batch])



    return audio_embeddings, description_embeddings, neg_description_embeddings


def evaluate_encodings(embeddings, labels, class_size, embedding_type, evaluation_type, experiment_type):
    """Evaluate a specific type of encoding (text or speech) for gender classification."""
    # Create dataset
    dataset = FeatureClassificationDataset(embeddings, labels)

    # Split dataset
    total_size = len(dataset)
    train_size = int(0.9 * total_size)
    test_size = total_size - train_size

    train_dataset, test_dataset = random_split(
        dataset, [train_size, test_size]
    )

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # Initialize model and training components
    model = SpeechFeatureClassifier(output_dim=class_size).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    training_losses, validation_accuracies = [], []
    best_val_acc = 0.0
    os.makedirs("../models/Evaluation_Classifiers", exist_ok=True)

    for epoch in range(NUM_EPOCHS):
        # Training phase
        model.train()
        total_train_loss = 0
        for batch_embeddings, batch_labels in train_loader:
            batch_embeddings, batch_labels = batch_embeddings.to(DEVICE), batch_labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(batch_embeddings)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        training_losses.append(avg_train_loss)

        # Validation phase
        model.eval()
        total_val_acc = 0
        with torch.no_grad():
            for val_embeddings, val_labels in test_loader:
                val_embeddings, val_labels = val_embeddings.to(DEVICE), val_labels.to(DEVICE)
                predictions = model.predict(val_embeddings)
                accuracy = (predictions == val_labels).float().mean()
                total_val_acc += accuracy.item()

        avg_val_acc = total_val_acc / len(test_loader)
        validation_accuracies.append(avg_val_acc)

        if avg_val_acc > best_val_acc:
            # Save the best mode
            best_val_acc = avg_val_acc
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
             "val_acc": best_val_acc}, f"../models/Evaluation_Classifiers/{experiment_type}_{embedding_type}_embedding_classifier_{evaluation_type}_labels.pt")
            print("New best model saved!")

        print(f"Epoch {epoch + 1} - Train Loss: {avg_train_loss:.4f}, Val Loss: {best_val_acc:.4f}")

    model_data = torch.load(f"../models/Evaluation_Classifiers/{experiment_type}_{embedding_type}_embedding_classifier_{evaluation_type}_labels.pt")
    model_data["training_losses"] = training_losses
    model_data["validation_accuracies"] = validation_accuracies
    torch.save(model_data, f"../models/Evaluation_Classifiers/{experiment_type}_{embedding_type}_embedding_classifier_{evaluation_type}_labels.pt")

    return best_val_acc

def run_gender_test():
    """Main evaluation pipeline."""
    print("#### RUNNING GENDER TEST ####")
    print("Loading dataset...")
    dataset = load_from_disk(ENRICHED_DATASET_PATH)
    dataset = dataset.cast_column("audio", Audio())
    
    # Load the trained DCCAE model
    model_state_dict = torch.load("../models/dccae_voice_text_best.pt", map_location=DEVICE)["model_state_dict"]
    model = create_dccae_model(state_dict=model_state_dict)

    
    model_description_embeddings = []
    model_speech_embeddings = []
    gender_labels = []
    
    def process_item(example):
        """Process a single dataset item."""
        try:
            # Convert audio to numpy array and preprocess
            description_embedding = example[DESCRIPTION_EMBEDDING_COLUMN]
            speech_embedding = example[SPEAKER_EMBEDDING_COLUMN]
            
            # Get embeddings
            with torch.no_grad():
                model_speech_embedding = model.encoders[0](torch.tensor(speech_embedding)).cpu().numpy()
                model_description_embedding = model.encoders[1](torch.tensor(description_embedding)).cpu().numpy()

            # Get gender label
            gender_label = 0 if example[GENDER_COLUMN] == "male" else 1

            return {
                'model_description_embedding': model_description_embedding,
                'model_speech_embedding': model_speech_embedding,
                'gender_label': gender_label,
                'success': True
            }

        except Exception as e:
            print(f"Error processing item: {e}")
            return {'success': False}
    
    print("Processing dataset...")
    for example in tqdm(dataset):
        result = process_item(example)
        if result['success']:
            model_description_embeddings.append(result['model_description_embedding'])
            model_speech_embeddings.append(result['model_speech_embedding'])
            gender_labels.append(result['gender_label'])
    
    text_embeddings = torch.tensor(model_description_embeddings).squeeze().to(DEVICE)
    speech_embeddings = torch.tensor(model_speech_embeddings).squeeze().to(DEVICE)
    gender_labels = torch.tensor(gender_labels).squeeze().long().to(DEVICE)
    
    # Create random baseline labels
    random_labels = torch.tensor([torch.randint(0, 2, ()).item() for _ in range(len(gender_labels))]).long().to(DEVICE)
    
    # Evaluate text encodings
    print("\n Evaluating on Correct Labels...")
    print("\nEvaluating Text Encodings...")
    text_accuracy = evaluate_encodings(text_embeddings, gender_labels, class_size=2, embedding_type="Text", evaluation_type="Correct", experiment_type="Gender")
    
    # Evaluate speech encodings
    print("\nEvaluating Speech Encodings...")
    speech_accuracy = evaluate_encodings(speech_embeddings, gender_labels, class_size=2, embedding_type="Speech", evaluation_type="Correct", experiment_type="Gender")
    
    # Evaluate random baseline
    print("\nEvaluating Random Baseline...")
    print("\nEvaluating Text Encodings with Random Labels...")
    random_text_accuracy = evaluate_encodings(text_embeddings, random_labels, class_size=2, embedding_type="Text", evaluation_type="Random", experiment_type="Gender")

    print("\nEvaluating Speech Encodings with Random Labels...")
    random_speech_accuracy = evaluate_encodings(speech_embeddings, random_labels, class_size=2, embedding_type="Speech", evaluation_type="Random", experiment_type="Gender")
    
    # Print final results
    print("\n=== Final Results ===")
    print(f"Text Encoding Accuracy: {text_accuracy:.4f}")
    print(f"Speech Encoding Accuracy: {speech_accuracy:.4f}")
    print(f"Random Labels Text Encoding Accuracy: {random_text_accuracy:.4f}")
    print(f"Random Labels Speech Encoding Accuracy: {random_speech_accuracy:.4f}")



def run_pitch_test():
    print("#### RUNNING PITCH TEST ####")
    print("Loading dataset...")
    dataset = load_from_disk(ENRICHED_DATASET_PATH)
    dataset = dataset.cast_column("audio", Audio())

    # Load the trained DCCAE model
    model_state_dict = torch.load("../models/dccae_voice_text_best.pt", map_location=DEVICE)["model_state_dict"]
    model = create_dccae_model(state_dict=model_state_dict)

    model_description_embeddings = []
    model_speech_embeddings = []
    pitch_labels = []

    def process_item(example):
        """Process a single dataset item."""
        try:
            # Convert audio to numpy array and preprocess
            description_embedding = example[DESCRIPTION_EMBEDDING_COLUMN]
            speech_embedding = example[SPEAKER_EMBEDDING_COLUMN]

            # Get embeddings
            with torch.no_grad():
                model_speech_embedding = model.encoders[0](torch.tensor(speech_embedding)).cpu().numpy()
                model_description_embedding = model.encoders[1](torch.tensor(description_embedding)).cpu().numpy()

            # Get gender label
            gender_label = 0 if example[PITCH_COLUMN] == "low" else 1 if example[PITCH_COLUMN] == "normal" else 2

            return {
                'model_description_embedding': model_description_embedding,
                'model_speech_embedding': model_speech_embedding,
                'pitch_label': gender_label,
                'success': True
            }

        except Exception as e:
            print(f"Error processing item: {e}")
            return {'success': False}

    print("Processing dataset...")
    for example in tqdm(dataset):
        result = process_item(example)
        if result['success']:
            model_description_embeddings.append(result['model_description_embedding'])
            model_speech_embeddings.append(result['model_speech_embedding'])
            pitch_labels.append(result['pitch_label'])

    text_embeddings = torch.tensor(model_description_embeddings).squeeze().to(DEVICE)
    speech_embeddings = torch.tensor(model_speech_embeddings).squeeze().to(DEVICE)
    pitch_labels = torch.tensor(pitch_labels).squeeze().long().to(DEVICE)

    # Create random baseline labels
    random_labels = torch.tensor([torch.randint(0, 3, ()).item() for _ in range(len(pitch_labels))]).long().to(DEVICE)

    # Evaluate text encodings
    print("\n Evaluating on Correct Labels...")
    print("\nEvaluating Text Encodings...")
    text_accuracy = evaluate_encodings(text_embeddings, pitch_labels, class_size=3, embedding_type="Text", evaluation_type="Correct",
                                       experiment_type="Pitch")

    # Evaluate speech encodings
    print("\nEvaluating Speech Encodings...")
    speech_accuracy = evaluate_encodings(speech_embeddings, pitch_labels, class_size=3, embedding_type="Speech",
                                         evaluation_type="Correct", experiment_type="Pitch")

    # Evaluate random baseline
    print("\nEvaluating Random Baseline...")
    print("\nEvaluating Text Encodings with Random Labels...")
    random_text_accuracy = evaluate_encodings(text_embeddings, random_labels, class_size=3, embedding_type="Text",
                                              evaluation_type="Random", experiment_type="Pitch")

    print("\nEvaluating Speech Encodings with Random Labels...")
    random_speech_accuracy = evaluate_encodings(speech_embeddings, random_labels, class_size=3, embedding_type="Speech",
                                                evaluation_type="Random", experiment_type="Pitch")

    # Print final results
    print("\n=== Final Results ===")
    print(f"Text Encoding Accuracy: {text_accuracy:.4f}")
    print(f"Speech Encoding Accuracy: {speech_accuracy:.4f}")
    print(f"Random Labels Text Encoding Accuracy: {random_text_accuracy:.4f}")
    print(f"Random Labels Speech Encoding Accuracy: {random_speech_accuracy:.4f}")







if __name__ == "__main__":
    run_pitch_test()

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils.class_weight import compute_class_weight

import streamlit as st
import pandas as pd
import numpy as np
from collections import Counter
import re


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@st.cache_resource
def train_model():
    df = pd.read_csv("recipes_fruit_veg.csv")
    df = df.dropna(subset=["name"])

    df["label"] = 1
    df.loc[df["category"] == "fruit", "label"] = 0

    texts = df["name"].tolist()
    labels = df["label"].tolist()

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts,
        labels,
        test_size=0.2,
        stratify=labels,
        random_state=42
    )

    def preprocess(text):
        text = text.lower()
        text = re.sub(r"[^a-zA-Z\s]", "", text)
        return text.split()

    train_tokens = [preprocess(t) for t in train_texts]
    val_tokens = [preprocess(t) for t in val_texts]

    word_counts = Counter(word for sent in train_tokens for word in sent)

    word2idx = {"<pad>": 0, "<unk>": 1}

    for word, count in word_counts.items():
        if count >= 2:
            word2idx[word] = len(word2idx)

    def encode(tokens):
        return [word2idx.get(w, word2idx["<unk>"]) for w in tokens]

    train_encoded = [encode(t) for t in train_tokens]
    val_encoded = [encode(t) for t in val_tokens]

    class TextDataset(Dataset):
        def __init__(self, texts, labels):
            self.texts = texts
            self.labels = labels

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            return {
                "text": torch.tensor(self.texts[idx], dtype=torch.long),
                "label": torch.tensor(self.labels[idx], dtype=torch.long)
            }

    def collate_fn(batch):
        texts = [item["text"] for item in batch]
        labels = torch.tensor([item["label"] for item in batch])

        max_len = max(len(t) for t in texts)
        padded = torch.zeros(len(texts), max_len, dtype=torch.long)

        for i, t in enumerate(texts):
            padded[i, :len(t)] = t

        return padded.to(device), labels.to(device)

    train_dataset = TextDataset(train_encoded, train_labels)
    val_dataset = TextDataset(val_encoded, val_labels)

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=collate_fn
    )

    class TextClassifier(nn.Module):
        def __init__(self, vocab_size, embed_dim, num_classes=2):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            self.fc = nn.Linear(embed_dim, num_classes)

        def forward(self, x):
            emb = self.embedding(x)

            mask = (x != 0).unsqueeze(-1)
            emb = emb * mask

            pooled = emb.sum(dim=1) / mask.sum(dim=1).clamp(min=1)

            return self.fc(pooled)

    model = TextClassifier(len(word2idx), 64, 2).to(device)

    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(train_labels),
        y=train_labels
    )

    weights = torch.tensor(weights, dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(5):
        model.train()

        for x, y in train_loader:
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

    model.eval()

    preds, actuals = [], []

    with torch.no_grad():
        for x, y in val_loader:
            output = model(x)
            pred = torch.argmax(output, dim=1)

            preds.extend(pred.cpu().numpy())
            actuals.extend(y.cpu().numpy())

    metrics = {
        "accuracy": accuracy_score(actuals, preds),
        "precision": precision_score(actuals, preds),
        "recall": recall_score(actuals, preds),
        "f1": f1_score(actuals, preds)
    }

    return model, word2idx, preprocess, metrics


model, word2idx, preprocess, metrics = train_model()


def encode(tokens):
    return [word2idx.get(w, word2idx["<unk>"]) for w in tokens]


def predict(text):
    tokens = preprocess(text)
    encoded = encode(tokens)

    if len(encoded) == 0:
        return "Please enter a valid recipe name."

    tensor = torch.tensor(encoded, dtype=torch.long).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)
        pred = torch.argmax(output, dim=1).item()

    return "vegetable" if pred == 1 else "fruit"


st.title("Fruit vs Vegetable Recipe Classifier")

st.write("Enter a recipe name and the PyTorch model will classify it.")

user_input = st.text_input("Recipe name", placeholder="Example: Blueberry muffins")

if st.button("Predict"):
    result = predict(user_input)
    st.success(f"Prediction: {result}")

st.subheader("Model Metrics")

st.write(f"Accuracy: {metrics['accuracy']:.4f}")
st.write(f"Precision: {metrics['precision']:.4f}")
st.write(f"Recall: {metrics['recall']:.4f}")
st.write(f"F1 Score: {metrics['f1']:.4f}")

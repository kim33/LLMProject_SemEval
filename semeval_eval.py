import torch
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification

# Load the trained model and tokenizer
model_path = './bert-topic-cls'  # Replace with your model's path
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(model_path)
model.eval()

# Input CSV file (contains text data without labels)
input_file = "./dataset/eng_dev.csv"  # Replace with your file path
output_file = "./dataset/eng_dev_result.csv"  # Replace with desired output file name

# Load the dataset
df = pd.read_csv(input_file)

# Emotion label mapping
simple_label_map = {0: "anger", 1: "fear", 2: "joy", 3: "sadness", 4: "surprise"}

# Function to predict emotions using Top-k or Probability Ranking
def predict_emotions(text, method="top-k", top_k=2, threshold=0.8):
    """Predict emotions for a given text using the specified method."""
    # Tokenize the input text
    inputs = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Get predictions
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.sigmoid(logits).squeeze(0)  # Sigmoid for multi-label classification

    if method == "top-k":
        # Top-k prediction
        top_k_indices = torch.topk(probs, k=top_k).indices
        predicted_labels = [simple_label_map[idx.item()] for idx in top_k_indices]

    elif method == "probability-ranking":
        # Probability ranking
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_prob = 0.0
        predicted_labels = []

        for prob, idx in zip(sorted_probs, sorted_indices):
            cumulative_prob += prob.item()
            predicted_labels.append(simple_label_map[idx.item()])
            if cumulative_prob >= threshold:
                break

    else:
        raise ValueError("Invalid method. Use 'top-k' or 'probability-ranking'.")

    # Create a binary label vector for output
    binary_labels = {emotion: 0 for emotion in simple_label_map.values()}
    for label in predicted_labels:
        binary_labels[label] = 1

    return binary_labels

# Iterate through the dataset and predict emotions
predicted_data = []

for _, row in df.iterrows():
    text = row["text"]
    # Choose the prediction method ("top-k" or "probability-ranking")
    method = "top-k"  # Change to "probability-ranking" if needed
    predictions = predict_emotions(text, method=method, top_k=2, threshold=0.8)
    predicted_data.append({**row, **predictions})

# Save the results to a new CSV file
predicted_df = pd.DataFrame(predicted_data)
predicted_df.to_csv(output_file, index=False)

print(f"Predictions saved to {output_file}")

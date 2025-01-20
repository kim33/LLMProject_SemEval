from datasets import load_dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import random
import os
import torch

# Function to set the seed for reproducibility
def set_seed(seed_value=42):
    """Set seed for reproducibility."""
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)  # if you are using multi-GPU.
    random.seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)

    # The below two lines are for deterministic algorithm behavior in CUDA
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Set the seed
set_seed()

# Load your dataset
train_dataset = load_dataset("csv", data_files={"train" : "./dataset/eng_train.csv"})
test_dataset = load_dataset("csv", data_files={"test" : "./dataset/eng_test.csv"})


# Initialize the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenization function with labels included
def tokenize_function(examples):
    return tokenizer(examples['text'], padding=True, truncation=True, max_length=512)

# Tokenizing the datasets
tokenized_train_dataset = train_dataset['train'].map(tokenize_function, batched=True)
tokenized_test_dataset = test_dataset['test'].map(tokenize_function, batched=True)

# Remove 'id' column after tokenization (optional)
tokenized_train_dataset = tokenized_train_dataset.remove_columns(["id"])
tokenized_test_dataset = tokenized_test_dataset.remove_columns(["id"])

# Prepare the labels: Convert emotion columns to a binary vector for multi-label classification
def process_labels(examples):
    labels = ['anger', 'fear', 'joy', 'sadness', 'surprise']
    label_vector = []
    for label in labels:
        # Check if the column exists and get its value
        if label in examples:
            value = examples[label]
            
            # Ensure the value is a single number (0 or 1)
            # check if it's a list of values, or a single value
            if isinstance(value, list) :
                label_vector.append(1 if label in value else 0)
            elif isinstance(value, (int, float)):
                label_vector.append(float(value))  # Convert to 0 or 1
            else:
                # Handle unexpected types by defaulting to 0 (no emotion)
                label_vector.append(float(0))
        else:
            # If the column is missing, default to 0
            print(f"Warning: Column {label} is missing in the input data!")
            label_vector.append(0)
    
    examples['labels'] = torch.tensor(label_vector)  # Convert label vector to tensor
    return examples

# Tokenize and add labels to the datasets
tokenized_train_dataset = tokenized_train_dataset.map(process_labels, batched=False)
tokenized_test_dataset = tokenized_test_dataset.map(process_labels, batched=False)

# Data collator for padding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Load the pre-trained BERT model for multi-label classification
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=5)

# Training arguments
training_args = TrainingArguments(
    output_dir='./results-bert-topic-cls',
    num_train_epochs=30,
    per_device_train_batch_size=10,
    per_device_eval_batch_size=5,
    warmup_steps=100,
    weight_decay=0.02,
    logging_dir='./logs',
    evaluation_strategy='epoch',  # Evaluate at the end of each epoch
    logging_steps=10,
    report_to="tensorboard",
)

# Compute metrics function for evaluation
def compute_metrics(p):
    predictions, labels = p
    
    predictions = np.argmax(predictions, axis=1)  # Get the index of the highest logit (class)
    
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    acc = accuracy_score(labels, predictions)
    return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}
# Trainer initialization
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_test_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()

# Save the model
output_dir = './bert-topic-cls'
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

# Evaluate the model
results = trainer.evaluate()

# Predictions and confusion matrix
predictions = trainer.predict(tokenized_test_dataset)
preds = torch.sigmoid(torch.tensor(predictions.predictions)) > 0.5  # Threshold at 0.5 to predict binary labels

# Confusion matrix and metrics
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

label_map = {
    'anger': 'anger',
    'fear': 'fear',
    'joy': 'joy',
    'sadness': 'sadness',
    'surprise': 'surprise'
}

# Flatten the predictions and labels for confusion matrix
flattened_preds = preds.numpy().flatten()
flattened_labels = predictions.label_ids.flatten()

cm = confusion_matrix(flattened_labels, flattened_preds)
labels = [label_map[label] for label in label_map]

# Plot confusion matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix with Label Names')
plt.show()


import torch

dev_dataset = load_dataset("csv", data_files={"dev" : "./dataset/eng_dev.csv"})

# Example sentence
sentence = "I was very shocked."

# Tokenize the sentence
inputs = tokenizer(sentence, padding=True, truncation=True, max_length=512, return_tensors="pt")

# Move inputs to the same device as the model
inputs = {k: v.to(model.device) for k, v in inputs.items()}

# Make prediction using softmax
model.eval()  # Set the model to evaluation mode
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits  # Get the logits from the model output

# Apply softmax to get class probabilities (optional, for understanding)
probs = torch.softmax(logits, dim=-1)

# Get the predicted class (index with highest probability)
predicted_class = torch.argmax(probs, dim=-1).item()

# Map the prediction index to the class name (if you have a label map)
label_map = {0: "anger", 1: "fear", 2: "joy", 3: "sadness", 4: "surprise"}
predicted_label = label_map[predicted_class]

print(f"Sentence: '{sentence}'")
print(f"Predicted Label: '{predicted_label}'")
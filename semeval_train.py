from datasets import load_dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import random
import os
import torch
import shutil

# Load your dataset
train_dataset = load_dataset("csv", data_files={"train" : "./dataset/eng_train.csv"})
test_dataset = load_dataset("csv", data_files={"test" : "./dataset/eng_test.csv"})

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=5, from_flax=False)
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
data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)

# Load the pre-trained BERT model for multi-label classification

# Training arguments
training_args = TrainingArguments(
    output_dir='./results-bert-topic-cls',
    num_train_epochs=35,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=5e-7,
    warmup_steps=100,
    weight_decay=0.05,
    evaluation_strategy='epoch',
    logging_strategy='epoch',
    lr_scheduler_type='constant'
)

# Compute metrics function for multi-label classification
def compute_metrics(p):
    predictions, labels = p
    
    # Apply sigmoid to the predictions for multi-label classification
    predictions = torch.sigmoid(torch.tensor(predictions)) > 0.5  # Threshold at 0.5 to predict binary labels
    
    # Flatten predictions and labels to calculate metrics across all labels
    predictions = predictions.numpy().flatten()
    labels = labels.flatten()
    
    # Compute precision, recall, f1, and accuracy
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='macro', zero_division=0)
    acc = accuracy_score(labels, predictions)
    
    return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}

# Trainer initialization with the new compute_metrics
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_test_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics
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
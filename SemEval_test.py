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
dev_dataset = load_dataset("csv", data_files={"dev" : "./dataset/eng_dev.csv"})

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
            if isinstance(value, (int, float)):
                label_vector.append(float(value))  # Convert to 0 or 1
            elif isinstance(value, list) and len(value) == 1:
                # If the value is a list with a single element, use it
                label_vector.append(float(value[0]))
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
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=5, problem_type="multi_label_classification")

# Training arguments
training_args = TrainingArguments(
    output_dir='./results-bert-topic-cls',
    num_train_epochs=3,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    evaluation_strategy='epoch',  # Evaluate at the end of each epoch
    logging_steps=10,
    report_to="tensorboard",
)

# Compute metrics function for evaluation
def compute_metrics(p):
    predictions, labels = p
    
    # Convert predictions from logits to probabilities (sigmoid) for multi-label classification
    predictions = torch.sigmoid(torch.tensor(predictions))  # Apply sigmoid activation
    
    # Convert to binary (0 or 1) predictions
    predictions = predictions > 0.5  # Convert probabilities to binary labels (0 or 1)
    
    # Cast labels to Long type if necessary (e.g., for classification tasks)
    labels = torch.tensor(labels, dtype=torch.long)  # Ensure labels are of type Long
    
    precision, recall, f1, _ = precision_recall_fscore_support(labels.numpy(), predictions.numpy(), average='weighted', zero_division=0)
    acc = accuracy_score(labels.numpy(), predictions.numpy())
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

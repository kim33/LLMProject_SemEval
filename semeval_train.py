from datasets import load_dataset
from transformers import XLMRobertaForSequenceClassification, XLMRobertaTokenizer, Trainer, TrainingArguments, DataCollatorWithPadding
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import random
import os
import torch

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

from datasets import load_dataset

# Load your dataset (single file for train/test)
dataset = load_dataset("csv", data_files={"data": "./dataset/eng_train.csv"})

# Split the dataset into train (80%) and test (20%)
split_datasets = dataset["data"].train_test_split(test_size=0.2, seed=42)

# Extract the train and test datasets
train_dataset = split_datasets["train"]
test_dataset = split_datasets["test"]

model = XLMRobertaForSequenceClassification.from_pretrained(
    'xlm-roberta-large', 
    num_labels=5, 
    from_flax=False, 
    problem_type="multi_label_classification", 
    hidden_dropout_prob=0.2
)

# Initialize the tokenizer
tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-large')

# Tokenization function with labels included
def tokenize_function(examples):
    return tokenizer(examples['text'], padding=True, truncation=True, max_length=512)

# Tokenize the datasets
tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True)

# Remove 'id' column after tokenization (optional)
if "id" in tokenized_train_dataset.column_names:
    tokenized_train_dataset = tokenized_train_dataset.remove_columns(["id"])
if "id" in tokenized_test_dataset.column_names:
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
    num_train_epochs=50,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=3e-5,
    warmup_steps=200,
    weight_decay=0.02,
    evaluation_strategy='epoch',
    logging_strategy='epoch',
    lr_scheduler_type='constant',
    report_to='none'
)

def compute_metrics(p):
    predictions, labels = p
    
    # Apply sigmoid to the predictions for multi-label classification
    probabilities = torch.sigmoid(torch.tensor(predictions))  # Convert logits to probabilities
    
    # Dynamically compute threshold (based on mean of probabilities)
    dynamic_thresholds = probabilities.mean(dim=1, keepdim=True)  # Use the mean as dynamic threshold
    
    # Select labels above the dynamic threshold
    selected_predictions = (probabilities > dynamic_thresholds).int()
    
    # Flatten predictions and labels to calculate metrics across all labels
    selected_predictions = selected_predictions.numpy().flatten()
    labels = labels.flatten()
    
    # Compute precision, recall, f1, and accuracy
    # Macro-averaging :  Assessing the model's performance on all labels equally, regardless of frequency.
    # Macro-averaging treats rare and frequent emotions equally, which is helpful if rare emotions are just as important as common ones.
    precision, recall, f1, _ = precision_recall_fscore_support(labels, selected_predictions, average='macro', zero_division=0)
    acc = accuracy_score(labels, selected_predictions)
    
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
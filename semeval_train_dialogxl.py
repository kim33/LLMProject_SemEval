from torch.utils.data import DataLoader
from transformers import XLMRobertaForSequenceClassification, XLMRobertaTokenizer, AutoModel, DataCollatorWithPadding
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch
import torch.nn as nn
from transformers import Trainer, TrainingArguments

# Define the combined model (RoBERTa and DialogXL) as before
class EmotionDetectionModel(nn.Module):
    def __init__(self, roberta_model, dialogxl_model, num_emotions=5):
        super(EmotionDetectionModel, self).__init__()
        self.roberta = roberta_model
        self.dialogxl = dialogxl_model
        self.fc = nn.Linear(roberta_model.config.hidden_size + dialogxl_model.config.hidden_size, num_emotions)

    def forward(self, input_ids, attention_mask):
        # Get RoBERTa outputs
        roberta_outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        roberta_hidden = roberta_outputs.last_hidden_state[:, 0, :]  # Take the [CLS] token
        
        # Get DialogXL outputs
        dialogxl_outputs = self.dialogxl(input_ids=input_ids, attention_mask=attention_mask)
        dialogxl_hidden = dialogxl_outputs.last_hidden_state[:, 0, :]  # Take the [CLS] token
        
        # Concatenate the hidden states from both models
        combined_output = torch.cat((roberta_hidden, dialogxl_hidden), dim=-1)
        
        # Emotion classification head
        emotion_logits = self.fc(combined_output)
        return emotion_logits

# Step 1: Set up the model and tokenizer for RoBERTa
roberta_model_name = 'xlm-roberta-large'
tokenizer = XLMRobertaTokenizer.from_pretrained(roberta_model_name)
roberta_model = XLMRobertaForSequenceClassification.from_pretrained(
        'xlm-roberta-large',
        num_labels=5,
        hidden_dropout_prob=0.2,
        problem_type="multi_label_classification")

# Initialize DialogXL model
dialogxl_model_name = 'xlnet-base-cased'  # Replace with DialogXL model if available
dialogxl_model = AutoModel.from_pretrained(dialogxl_model_name)


# Step 3: Load the dataset from Hugging Face Dataset library
from datasets import load_dataset

# Load dataset
dataset = load_dataset("csv", data_files={"data": "./dataset/eng_train.csv"})
split_datasets = dataset["data"].train_test_split(test_size=0.2, seed=42)
train_dataset = split_datasets["train"]
test_dataset = split_datasets["test"]

# Tokenization function
def tokenize_function(examples):
    return tokenizer(examples['text'], padding=True, truncation=True, max_length=512)

# Tokenize the datasets
tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)
# Process labels for multi-label classification
def process_labels(examples):
    labels = ['anger', 'fear', 'joy', 'sadness', 'surprise']
    label_vector = []
    for label in labels:
        if label in examples:
            value = examples[label]
            if isinstance(value, list):
                label_vector.append(1 if label in value else 0)
            elif isinstance(value, (int, float)):
                label_vector.append(float(value))
            else:
                label_vector.append(0)
        else:
            label_vector.append(0)
    examples['labels'] = torch.tensor(label_vector)
    return examples

tokenized_train_dataset = tokenized_train_dataset.map(process_labels, batched=False)
tokenized_test_dataset = tokenized_test_dataset.map(process_labels, batched=False)

# Compute metrics
def compute_metrics(p):
    predictions, labels = p
    probabilities = torch.sigmoid(torch.tensor(predictions))
    dynamic_thresholds = probabilities.mean(dim=1, keepdim=True)
    selected_predictions = (probabilities > dynamic_thresholds).int()
    selected_predictions = selected_predictions.numpy().flatten()
    labels = labels.flatten()
    precision, recall, f1, _ = precision_recall_fscore_support(labels, selected_predictions, average='macro', zero_division=0)
    acc = accuracy_score(labels, selected_predictions)
    return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}

# Step 4: Use the Hugging Face Trainer API
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=200,
    weight_decay=0.01,
    logging_dir='./logs',
    report_to='none'
)

# Step 5: Initialize the Trainer with model and datasets
model = EmotionDetectionModel(roberta_model, dialogxl_model)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_test_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Step 6: Train the model
trainer.train()

# Save the best model
output_dir = './final-dialogxl-model'
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

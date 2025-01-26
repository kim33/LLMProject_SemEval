import optuna
from transformers import TrainingArguments, Trainer
from datasets import load_dataset
from transformers import XLMRobertaForSequenceClassification, XLMRobertaTokenizer, DataCollatorWithPadding
import numpy as np
import random
import os
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import re
from openai import OpenAI

# Set the seed for reproducibility
def set_seed(seed_value=42):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)  # if you are using multi-GPU.
    random.seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()

# Load dataset
dataset = load_dataset("csv", data_files={"data": "./dataset/eng_train.csv"})
split_datasets = dataset["data"].train_test_split(test_size=0.2, seed=42)
train_dataset = split_datasets["train"]
test_dataset = split_datasets["test"]

# Initialize tokenizer
tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-large')

# Tokenization function
def tokenize_function(examples):
    return tokenizer(examples['text'], padding=True, truncation=True, max_length=512)

tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True)

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

data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)
client = OpenAI(api_key = 'my_api_key')


def call_gpt4_as_judge(predictions, labels):
    prompt = "Evaluate the following predictions against their ground truth labels for emotion recognition. Do not add the result to the final answer.\n"
    prompt += "Calculate score between 0 and 5 for each pair, indicating how well the prediction matches the label. Do not add the result to the final answer\n\n"

    # Create pairs of predictions and labels for the prompt
    pairs = [
        f"Prediction: {pred}, Ground Truth: {label}" 
        for pred, label in zip(predictions, labels)
    ]
    prompt += "\n".join(pairs)
    
    # Add instructions for evaluation
    prompt += "\n\n Calculate and return an average score of all pairs in a single number only as a final answer."

    # Call OpenAI API
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an expert evaluator for emotion recognition tasks. Your job is to calculate and return only the average score of the all input pairs"},
            {"role": "user", "content": prompt}
        ],
        max_tokens=1500,
        temperature=0.7
    )
    # Extract and parse GPT-4's score (assume it returns a JSON-like structure with scores)
    gpt_score = response.choices[0].message.content.strip()
    overall_score = re.findall(r'\d+\.\d+', gpt_score)
    print(overall_score)
    return overall_score

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
    gpt4_score = call_gpt4_as_judge(selected_predictions.tolist(), labels.tolist())[0]
    f1 = float(f1)
    gpt4_score = float(gpt4_score) / 5
    objective_value = (f1 + gpt4_score)/2 #equal weights applied to f1 and gpt_score
    return {'combined_score': objective_value, 'accuracy': acc, 'f1': f1, 'gp4t_score': gpt4_score, 'precision': precision, 'recall': recall}

# Optuna objective function
def objective(trial):
    # Define hyperparameters
    learning_rate = trial.suggest_loguniform("learning_rate", 5e-5, 3e-3)
    dropout_rate = trial.suggest_uniform("dropout_rate", 0.1, 0.3)
    batch_size = trial.suggest_categorical("batch_size", [16, 32])
    num_train_epochs = trial.suggest_int("num_train_epochs", 1, 2)
    
    # Model with trial parameters
    model = XLMRobertaForSequenceClassification.from_pretrained(
        'xlm-roberta-large',
        num_labels=5,
        hidden_dropout_prob=dropout_rate,
        problem_type="multi_label_classification"
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir='./optuna-results',
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_steps=200,
        weight_decay=0.02,
        evaluation_strategy='epoch',
        logging_strategy='epoch',
        lr_scheduler_type='linear',
        save_strategy='no',
        report_to='none'
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_test_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )
    
    # Train and evaluate
    trainer.train()
    eval_results = trainer.evaluate()
    
    # Use F1-score as the metric to optimize
    return eval_results["eval_combined_score"]

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=1)

# Print best hyperparameters
print("Best hyperparameters:", study.best_params)

# Train with best hyperparameters
best_params = study.best_params
model = XLMRobertaForSequenceClassification.from_pretrained(
    'xlm-roberta-large',
    num_labels=5,
    hidden_dropout_prob=best_params["dropout_rate"],
    problem_type="multi_label_classification"
)
training_args = TrainingArguments(
    output_dir='./final-results',
    num_train_epochs=best_params["num_train_epochs"],
    per_device_train_batch_size=best_params["batch_size"],
    per_device_eval_batch_size=best_params["batch_size"],
    learning_rate=best_params["learning_rate"],
    warmup_steps=200,
    weight_decay=0.02,
    evaluation_strategy='epoch',
    logging_strategy='epoch',
    lr_scheduler_type='linear',
    save_total_limit=1,
    report_to='none'
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_test_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)
trainer.train()

# Save the best model
output_dir = './final-optuna-model'
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)


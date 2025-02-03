import optuna
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback, pipeline
from datasets import load_dataset, concatenate_datasets, Dataset
from transformers import RobertaForSequenceClassification, RobertaTokenizer, DataCollatorWithPadding
import numpy as np
import random
import os
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import seaborn as sns
import shap

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
train_dataset = dataset["data"]
final_validation = []
# Initialize tokenizer
tokenizer = RobertaTokenizer.from_pretrained('roberta-large')

# Tokenization function
def tokenize_function(examples):
    return tokenizer(examples['text'], padding=True, truncation=True, max_length=512)

tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
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

data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)

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

# Optuna objective function
def objective(trial):
    # Define hyperparameters
    learning_rate = trial.suggest_loguniform("learning_rate", 5e-5, 1e-4)
    dropout_rate = trial.suggest_uniform("dropout_rate", 0.1, 0.2)
    batch_size = trial.suggest_categorical("batch_size", [8, 16])
    num_train_epochs = trial.suggest_int("num_train_epochs", 3, 6)
    
    # Model with trial parameters
    model = RobertaForSequenceClassification.from_pretrained(
        'roberta-large',
        num_labels=5,
        hidden_dropout_prob=dropout_rate,
        problem_type="multi_label_classification"
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir='./roberta-5fold-final-shap-test',
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_steps=200,
        weight_decay=0.02,
        evaluation_strategy='epoch',
        logging_strategy='epoch',
        lr_scheduler_type='linear',
        save_strategy='epoch',
        report_to='none',
        load_best_model_at_end=True,  # Ensure that the best model is loaded at the end of training
        metric_for_best_model="eval_f1",  # Use F1 score to determine the best model
        greater_is_better=True,  # We want to maximize the F1 score
    )
    # K-Fold Cross-Validation
    k_folds = 10
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    all_metrics = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(tokenized_train_dataset)):
        print(f"Training fold {fold + 1}/{k_folds}...")
        
        # Split the data into train and validation sets for this fold
        train_split = tokenized_train_dataset.select(train_idx)
        val_split = tokenized_train_dataset.select(val_idx)
        final_validation.append(val_split)
        # Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_split,
            eval_dataset=val_split,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        )
        
        # Train and evaluate
        trainer.train()
        eval_results = trainer.evaluate()
        all_metrics.append(eval_results)

    # Average metrics across folds
    avg_f1 = np.mean([metrics['eval_f1'] for metrics in all_metrics])
    print(f"Average F1 score for this trial: {avg_f1}")
    
    return avg_f1

# Run Optuna study
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=1)

# Print best hyperparameters
print("Best hyperparameters:", study.best_params)

# Train with best hyperparameters
best_params = study.best_params
model = RobertaForSequenceClassification.from_pretrained(
    'roberta-large',
    num_labels=5,
    hidden_dropout_prob=best_params["dropout_rate"],
    problem_type="multi_label_classification"
)
training_args = TrainingArguments(
    output_dir='./results/roberta-10fold-final-matrix',
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
    save_strategy='epoch',
    report_to='none',
    load_best_model_at_end=True,  # Ensure that the best model is loaded at the end of training
    metric_for_best_model="eval_f1",  # Use F1 score to determine the best model
    greater_is_better=True,  # We want to maximize the F1 score
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_train_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)
trainer.train()

# Save the best model
output_dir = './model/roberta-5fold-final-shap-test'
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)


combined_val_data = concatenate_datasets(final_validation)

combined_val_df = combined_val_data.to_pandas()
print(f"Combined validation dataset has {len(combined_val_df)} samples before removing duplicates.")
combined_val_df = combined_val_df.drop_duplicates(subset=['text'])
print(f"Combined validation dataset has {len(combined_val_df)} samples after removing duplicates.")
combined_val_data = Dataset.from_pandas(combined_val_df)

# Evaluate the best model and draw confusion matrix
best_model = RobertaForSequenceClassification.from_pretrained(
    output_dir,
    num_labels=5,
    hidden_dropout_prob=best_params["dropout_rate"],
    problem_type="multi_label_classification"
)

trainer_best = Trainer(
    model=best_model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_train_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)

eval_results = trainer_best.evaluate()
print("Evaluation results for the best model:", eval_results)

class_names = ['anger', 'fear', 'joy', 'sadness', 'surprise']
# Get predictions
predictions = trainer_best.predict(combined_val_data)
y_true = combined_val_data
y_pred = np.argmax(predictions.predictions, axis=1)

if "labels" in combined_val_data.column_names:
    true_labels = np.array(combined_val_data["labels"])
    y_true = np.argmax(true_labels, axis=1)
else:
    raise ValueError("The combined validation dataset does not have a 'labels' field.")

def draw_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    # Save the figure before showing
    plt.savefig("roberta_10fold_final_matrix.png", dpi=300, bbox_inches="tight")
    plt.show()

# Draw the confusion matrix
draw_confusion_matrix(y_true, y_pred, class_names)
#predictions.to_csv("./dataset/test_prediction.csv", index = False)


classifier = pipeline(
    "text-classification",
    model=model,
    tokenizer=tokenizer,
    device=0,  
    return_all_scores=True 
)
model.eval()


# Example texts from your test set (replace with actual test dataset text)
test_texts = combined_val_data.select(range(3))
#test_texts = [tokenizer.decode(x['input_ids'], skip_special_tokens=True) for x in tokenized_test_dataset.select(range(3))]
id2label = model.config.id2label
label_mapping = {
        'LABEL_0': 'anger',
        'LABEL_1': 'fear',
        'LABEL_2': 'joy',
        'LABEL_3': 'sadness',
        'LABEL_4': 'surprise'
}


original_label = [label_mapping[id2label[i]] for i in range(len(id2label))]

explainer = shap.Explainer(classifier, output_names=original_label)
shap_values=explainer(test_texts[:3])

#1.Visualize the impact on a single class

class_to_visualize = "anger"

class_index = original_label.index(class_to_visualize)

shap_values_single_class = shap_values[..., class_index]

# Save the visualization to an HTML file
file_name = f'SHAP_{class_to_visualize}.html'
with open(file_name, 'w') as file:
    file.write(shap.plots.text(shap_values_single_class, display=False))

print(f"SHAP visualization saved to {file_name}")


#2.Plotting the top words impacting a specific class

# Select the class to visualize (e.g., "joy")
class_to_visualize = "joy"

class_index = original_label.index(class_to_visualize)
shap_values_mean = shap_values[:, :, class_index].mean(0)

# Create the SHAP bar plot
plt.figure(figsize=(8, 6))  
shap.plots.bar(shap_values_mean, show=False) 

# Save as an image
file_name = f"SHAP_{class_to_visualize}_bar.png"
plt.savefig(file_name, bbox_inches='tight', dpi=300) 

print(f"SHAP bar chart saved as {file_name}")
shap.save_html(f"SHAP_{class_to_visualize}_force.html", shap.plots.force(shap_values[:, :, class_index]))

# Function to save SHAP visualization to HTML
def save_shap_html(filename, shap_values):
    with open(filename, 'w') as file:
        file.write(shap.plots.text(shap_values, display=False))

# Save the SHAP visualization
save_shap_html('SHAP_log_odds.html', shap_values)

print("SHAP explanations saved to SHAP_log_odds.html")


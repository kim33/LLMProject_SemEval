from datasets import load_dataset
from transformers import XLMRobertaForSequenceClassification, XLMRobertaTokenizer, Trainer, TrainingArguments, DataCollatorWithPadding
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import random
import os
import torch
import shap
from IPython.display import display


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

# Load your dataset (single file for train/test)
dataset = load_dataset("csv", data_files={"data": "./dataset/eng_train.csv"})

# Split the dataset into train (80%) and test (20%)
split_datasets = dataset["data"].train_test_split(test_size=0.2, seed=42)

# Extract the train and test datasets
train_dataset = split_datasets["train"]
test_dataset = split_datasets["test"]

# Initialize the model
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
'''
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

# Training arguments
training_args = TrainingArguments(
    output_dir='./results-bert-topic-cls',
    num_train_epochs=1,
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
    label_mapping = {
        0: 'anger',
        1: 'fear',
        2: 'joy',
        3: 'adness',
        4: 'urprise'
    }
    predicted_labels = [label_mapping[i] for i, label in enumerate(selected_predictions) if label == 1]
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
    compute_metrics=compute_metrics, 
)

# Train the model
trainer.train()

# Save the model
output_dir = './RoBERTa-emotion-cls'
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
'''
# Now, using transformers.pipeline for inference:
from transformers import pipeline
output_dir = './final-optuna-model'
# Load the trained model and tokenizer
model = XLMRobertaForSequenceClassification.from_pretrained(output_dir)
tokenizer = XLMRobertaTokenizer.from_pretrained(output_dir)

# Define a pipeline for emotion classification (multi-label classification)
classifier = pipeline(
    "text-classification",
    model=model,
    tokenizer=tokenizer,
    device=0,  # If using GPU; set to -1 for CPU
    return_all_scores=True  # Ensure we return all scores (probabilities) for each emotion class
)
model.eval()


# Example texts from your test set (replace with actual test dataset text)
test_texts = test_dataset.select(range(3))
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



import shap
import matplotlib.pyplot as plt

# Select the class to visualize (e.g., "joy")
class_to_visualize = "joy"

# Get the index of the class in the label mapping
class_index = original_label.index(class_to_visualize)

# Compute mean SHAP values across all examples for the selected class
shap_values_mean = shap_values[:, :, class_index].mean(0)

# Create the SHAP bar plot
plt.figure(figsize=(8, 6))  # Set figure size
shap.plots.bar(shap_values_mean, show=False)  # Generate bar plot without displaying

# Save as an image
file_name = f"SHAP_{class_to_visualize}_bar.png"
plt.savefig(file_name, bbox_inches='tight', dpi=300)  # Save the figure

print(f"SHAP bar chart saved as {file_name}")

# Show the plot (optional)
plt.show()

# save an HTML file, the only way is to use a force plot:
shap.save_html(f"SHAP_{class_to_visualize}_force.html", shap.plots.force(shap_values[:, :, class_index]))


file = open('RoBERTuna_30.html','w')
file.write(shap.plots.text(shap_values, display=False))
file.close

#shap_values = explainer(test_texts[:3])

#file = open('RoBERTuna_30.html','w')
#file.write(shap.plots.text(shap_values, display=False))
#file.close

# Print the predictions with the scores
#for text, pred in zip(test_texts, predictions):
#    print(f"Text: {text}")
#    for label in pred:
#        print(f"  {label['label']}: {label['score']:.4f}")

# Note: The output will show the scores for each emotion (anger, fear, joy, sadness, surprise) for each input text.
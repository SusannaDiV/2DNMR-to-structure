import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from sklearn.metrics import mean_squared_error, accuracy_score
import numpy as np
import wandb
import argparse

# Initialize Weights & Biases project
wandb.init(project="molecule-generation", name="Formula Conditioned SMILES Generation")

# Argument parser for input files and directories
parser = argparse.ArgumentParser(description="Train T5 model for SMILES generation from NMR data")
parser.add_argument("--train_src", type=str, required=True, help="Path to training source data (NMR data)")
parser.add_argument("--train_tgt", type=str, required=True, help="Path to training target data (SMILES)")
parser.add_argument("--test_src", type=str, required=True, help="Path to testing source data (NMR data)")
parser.add_argument("--test_tgt", type=str, required=True, help="Path to testing target data (SMILES)")
parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the model and tokenizer")

args = parser.parse_args()

class MoleculeDataset(Dataset):
    def __init__(self, src_texts, tgt_texts, tokenizer, max_length=512):
        self.src_texts = src_texts
        self.tgt_texts = tgt_texts
        self.tokenizer = tokenizer
        self.max_length = max_length

        if len(self.src_texts) != len(self.tgt_texts):
            raise ValueError(f"Source and target texts have different lengths: {len(self.src_texts)} vs {len(self.tgt_texts)}")

    def __len__(self):
        return len(self.src_texts)

    def __getitem__(self, idx):
        src_line = self.src_texts[idx].strip()
        target_text = self.tgt_texts[idx].strip()

        formula, nmr_data = src_line.split(' ', 1)

        # Conditional Generation on the molecular formula: combine the molecular formula with the NMR data as the input text
        conditioned_input_text = f"{formula} | {nmr_data}"

        # Tokenize the conditioned input text and target text
        input_encoding = self.tokenizer(conditioned_input_text, padding='max_length', truncation=True, max_length=self.max_length, return_tensors="pt")
        target_encoding = self.tokenizer(target_text, padding='max_length', truncation=True, max_length=self.max_length, return_tensors="pt")

        input_ids = input_encoding['input_ids'].squeeze()
        attention_mask = input_encoding['attention_mask'].squeeze()
        labels = target_encoding['input_ids'].squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

    def collate_fn(self, batch):
        batch = [item for item in batch if item is not None]
        return torch.utils.data.dataloader.default_collate(batch)

def load_data(src_path, tgt_path):
    with open(src_path, 'r') as src_file, open(tgt_path, 'r') as tgt_file:
        src_lines = src_file.readlines()
        tgt_lines = tgt_file.readlines()

        if len(src_lines) != len(tgt_lines):
            raise ValueError(f"Source and target files have different number of lines: {len(src_lines)} vs {len(tgt_lines)}")
        
    return src_lines, tgt_lines

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print(f"CUDA is available. Device count: {torch.cuda.device_count()}")
else:
    print("CUDA is not available.")

# Load data
train_src_texts, train_tgt_texts = load_data(args.train_src, args.train_tgt)
test_src_texts, test_tgt_texts = load_data(args.test_src, args.test_tgt)

print(f"Training data - src: {len(train_src_texts)}, tgt: {len(train_tgt_texts)}")
print(f"Testing data - src: {len(test_src_texts)}, tgt: {len(test_tgt_texts)}")

# Load tokenizer and model
tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small").to(device)

# Prepare datasets and dataloaders
train_dataset = MoleculeDataset(train_src_texts, train_tgt_texts, tokenizer)
test_dataset = MoleculeDataset(test_src_texts, test_tgt_texts, tokenizer)
train_dataloader = DataLoader(train_dataset, batch_size=2, collate_fn=train_dataset.collate_fn, pin_memory=True)
test_dataloader = DataLoader(test_dataset, batch_size=2, collate_fn=test_dataset.collate_fn, pin_memory=True)

# Define training arguments
training_args = TrainingArguments(
    output_dir=args.output_dir,          
    num_train_epochs=3,              
    per_device_train_batch_size=2,  
    save_steps=10_000,               
    save_total_limit=2,              
    evaluation_strategy="epoch",     
    fp16=True, 
)

# Initialize the Trainer
trainer = Trainer(
    model=model,                         
    args=training_args,                  
    train_dataset=train_dataset,         
    eval_dataset=test_dataset,           
    data_collator=train_dataset.collate_fn  
)

# Train the model
trainer.train()

# Save the model and tokenizer
model.save_pretrained(f'{args.output_dir}/model')
tokenizer.save_pretrained(f'{args.output_dir}/tokenizer')

def evaluate_model(model, dataloader, tokenizer, device, max_samples=100):
    model.eval()
    predictions = []
    references = []
    mse_scores = []
    counter = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, num_return_sequences=1, num_beams=10)

            decoded_preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            pred_token_ids = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(pred)) for pred in decoded_preds]
            labels = torch.where(labels != -100, labels, torch.tensor(tokenizer.pad_token_id).to(labels.device))
            label_token_ids = labels.cpu().numpy().tolist()

            decoded_labels = tokenizer.batch_decode(label_token_ids, skip_special_tokens=True)

            predictions.extend(decoded_preds)
            references.extend(decoded_labels)

            for pred, ref in zip(decoded_preds, decoded_labels):
                print(f"Prediction: {pred}")
                print(f"Reference: {ref}")
                print(f"Match: {pred == ref}")
                if pred != ref:
                    for i, (p, r) in enumerate(zip(pred, ref)):
                        if p != r:
                            print(f"Mismatch at position {i}: predicted '{p}' vs reference '{r}'")

            for pred, ref in zip(pred_token_ids, label_token_ids):
                min_len = min(len(pred), len(ref))
                mse_scores.append(mean_squared_error(ref[:min_len], pred[:min_len]))

            counter += len(batch["input_ids"])
            if counter >= max_samples:
                break

    accuracy = accuracy_score(references, predictions)
    mse = np.mean(mse_scores)
    exact_matches = np.mean([1 if pred == ref else 0 for pred, ref in zip(predictions, references)])
    top1 = np.mean([1 if ref in pred[:1] else 0 for pred, ref in zip(predictions, references)])
    top5 = np.mean([1 if ref in pred[:5] else 0 for pred, ref in zip(predictions, references)])
    top10 = np.mean([1 if ref in pred[:10] else 0 for pred, ref in zip(predictions, references)])

    return {
        "accuracy": accuracy,
        "mse": mse,
        "exact_match": exact_matches,
        "top1": top1,
        "top5": top5,
        "top10": top10,
    }

metrics = evaluate_model(model, test_dataloader, tokenizer, device, max_samples=100)

wandb.log(metrics)

print("Evaluation Metrics:")
for key, value in metrics.items():
    print(f"{key}: {value}")

wandb.finish()

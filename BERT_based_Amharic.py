# ===================================================
# AMHARIC EXTRACTIVE QA SYSTEM WITH BERT (FIXED)
# ===================================================

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    
    get_scheduler
)
from torch.optim import AdamW
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import re
import os

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ===================================================
# 1. CONFIGURATION
# ===================================================
class Config:
    MODEL_NAME = "rasyosef/bert-medium-amharic"
    DATA_PATH = "/content/drive/MyDrive/PHD folder/QA dataset/AmhQA25442.csv"
    MAX_LEN = 384
    DOC_STRIDE = 128
    BATCH_SIZE = 16
    EPOCHS = 5
    LEARNING_RATE = 3e-5
    WARMUP_RATIO = 0.1
    WEIGHT_DECAY = 0.01
    RANDOM_SEED = 42
    VALIDATION_SPLIT = 0.2
    NUM_WORKERS = 0          # Set to 0 to avoid multiprocessing issues in some environments
    USE_FP16 = True if device == 'cuda' else False

config = Config()
torch.manual_seed(config.RANDOM_SEED)

# ===================================================
# 2. LOAD AND PREPROCESS DATA
# ===================================================
print("\nLoading dataset...")
df = pd.read_csv(config.DATA_PATH)
df = df.fillna('')

# Ensure correct column names
if 'context' not in df.columns:
    raise KeyError("Column 'context' not found.")
if 'Question' not in df.columns:
    raise KeyError("Column 'Question' not found.")
if 'Answer' not in df.columns:
    raise KeyError("Column 'Answer' not found.")

df['context'] = df['context'].astype(str)
df['Question'] = df['Question'].astype(str)
df['Answer'] = df['Answer'].astype(str)
df = df[(df['context'].str.strip() != '') & (df['Question'].str.strip() != '')]
print(f"Dataset size after cleaning: {len(df)}")

# ===================================================
# 3. ANSWER SPAN DETECTION (robust)
# ===================================================
def tokenize_amharic(text):
    if not isinstance(text, str):
        return []
    text = re.sub(r'[^\u1200-\u137F\s]', '', text)
    return text.strip().split()

def find_answer_span(context, answer):
    context_norm = re.sub(r'\s+', ' ', context).strip()
    answer_norm = re.sub(r'\s+', ' ', answer).strip()
    
    # Exact match
    start_pos = context_norm.find(answer_norm)
    if start_pos != -1:
        return start_pos, start_pos + len(answer_norm)
    
    # Case-insensitive
    start_pos = context_norm.lower().find(answer_norm.lower())
    if start_pos != -1:
        return start_pos, start_pos + len(answer_norm)
    
    # Token-based matching
    ctx_tokens = tokenize_amharic(context_norm)
    ans_tokens = tokenize_amharic(answer_norm)
    for i in range(len(ctx_tokens) - len(ans_tokens) + 1):
        if ctx_tokens[i:i+len(ans_tokens)] == ans_tokens:
            before = ' '.join(ctx_tokens[:i])
            start_char = len(before) + (1 if i > 0 else 0)
            end_char = start_char + len(' '.join(ans_tokens))
            return start_char, end_char
    
    return 0, 0  # fallback

# Apply span detection
print("\nDetecting answer spans...")
spans = []
for idx, row in tqdm(df.iterrows(), total=len(df)):
    start, end = find_answer_span(row['context'], row['Answer'])
    spans.append((start, end))

df['answer_start'] = [s[0] for s in spans]
df['answer_end'] = [s[1] for s in spans]
df = df[(df['answer_start'] >= 0) & (df['answer_end'] > df['answer_start'])]
print(f"Dataset size after span validation: {len(df)}")

# ===================================================
# 4. LOAD TOKENIZER AND MODEL
# ===================================================
print(f"\nLoading tokenizer and model: {config.MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
# Ensure pad_token is set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else '[PAD]'
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)

model = AutoModelForQuestionAnswering.from_pretrained(config.MODEL_NAME)
model.to(device)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# ===================================================
# 5. DATASET CLASS (with stride, but we keep raw examples)
# ===================================================
class AmharicQADataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len, doc_stride):
        self.dataframe = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.doc_stride = doc_stride
        self.examples = []
        self._prepare_examples()
    
    def _prepare_examples(self):
        for idx, row in tqdm(self.dataframe.iterrows(), total=len(self.dataframe), desc="Tokenizing"):
            context = row['context']
            question = row['Question']
            start_char = row['answer_start']
            end_char = row['answer_end']
            
            encoding = self.tokenizer(
                question,
                context,
                max_length=self.max_len,
                truncation="only_second",
                stride=self.doc_stride,
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                padding=False,
            )
            
            for i, offsets in enumerate(encoding["offset_mapping"]):
                start_token_idx = None
                end_token_idx = None
                for token_idx, (token_start, token_end) in enumerate(offsets):
                    if token_start is None or token_end is None:
                        continue
                    if start_char >= token_start and start_char < token_end:
                        start_token_idx = token_idx
                    if end_char > token_start and end_char <= token_end:
                        end_token_idx = token_idx
                        break
                
                if start_token_idx is not None and end_token_idx is not None:
                    self.examples.append({
                        "input_ids": encoding["input_ids"][i],
                        "attention_mask": encoding["attention_mask"][i],
                        "start_positions": start_token_idx,
                        "end_positions": end_token_idx,
                    })
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        ex = self.examples[idx]
        return {
            "input_ids": torch.tensor(ex["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(ex["attention_mask"], dtype=torch.long),
            "start_positions": torch.tensor(ex["start_positions"], dtype=torch.long),
            "end_positions": torch.tensor(ex["end_positions"], dtype=torch.long),
        }

# ===================================================
# 6. CUSTOM COLLATE FUNCTION (FIXES THE RUNTIME ERROR)
# ===================================================
def collate_fn(batch):
    input_ids = [item["input_ids"] for item in batch]
    attention_mask = [item["attention_mask"] for item in batch]
    start_positions = torch.tensor([item["start_positions"] for item in batch])
    end_positions = torch.tensor([item["end_positions"] for item in batch])
    
    # Pad sequences to the maximum length in this batch
    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask_padded = pad_sequence(attention_mask, batch_first=True, padding_value=0)
    
    return {
        "input_ids": input_ids_padded,
        "attention_mask": attention_mask_padded,
        "start_positions": start_positions,
        "end_positions": end_positions,
    }

# ===================================================
# 7. SPLIT DATA AND CREATE DATALOADERS
# ===================================================
train_df, val_df = train_test_split(
    df, test_size=config.VALIDATION_SPLIT, random_state=config.RANDOM_SEED
)
print(f"Training examples: {len(train_df)}")
print(f"Validation examples: {len(val_df)}")

train_dataset = AmharicQADataset(train_df, tokenizer, config.MAX_LEN, config.DOC_STRIDE)
val_dataset = AmharicQADataset(val_df, tokenizer, config.MAX_LEN, config.DOC_STRIDE)

train_loader = DataLoader(
    train_dataset,
    batch_size=config.BATCH_SIZE,
    shuffle=True,
    num_workers=config.NUM_WORKERS,
    collate_fn=collate_fn,
    pin_memory=True if device == 'cuda' else False
)

val_loader = DataLoader(
    val_dataset,
    batch_size=config.BATCH_SIZE,
    shuffle=False,
    num_workers=config.NUM_WORKERS,
    collate_fn=collate_fn,
    pin_memory=True if device == 'cuda' else False
)

print(f"Number of training batches: {len(train_loader)}")
print(f"Number of validation batches: {len(val_loader)}")

# ===================================================
# 8. METRICS: NORMALIZATION, EM, F1
# ===================================================
def normalize_answer(s):
    s = s.lower()
    s = re.sub(r'[^\w\s\u1200-\u137F]', '', s)
    return s.strip()

def compute_em_f1(pred_answer, true_answer):
    pred_norm = normalize_answer(pred_answer)
    true_norm = normalize_answer(true_answer)
    em = int(pred_norm == true_norm)
    
    pred_tokens = pred_norm.split()
    true_tokens = true_norm.split()
    if not pred_tokens and not true_tokens:
        f1 = 1.0
    elif not pred_tokens or not true_tokens:
        f1 = 0.0
    else:
        common = set(pred_tokens) & set(true_tokens)
        prec = len(common) / len(pred_tokens)
        rec = len(common) / len(true_tokens)
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return em, f1

def evaluate_model(model, val_dataframe, tokenizer, device):
    """
    Evaluate on the original validation dataframe (without stride).
    This is simpler and more accurate.
    """
    model.eval()
    em_scores = []
    f1_scores = []
    
    for idx, row in tqdm(val_dataframe.iterrows(), total=len(val_dataframe), desc="Evaluating"):
        context = row['context']
        question = row['Question']
        true_answer = row['Answer']
        
        # Tokenize without stride (single chunk)
        inputs = tokenizer(
            question,
            context,
            max_length=config.MAX_LEN,
            truncation="only_second",
            return_tensors="pt"
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            start_logits = outputs.start_logits[0].cpu().numpy()
            end_logits = outputs.end_logits[0].cpu().numpy()
            start_idx = np.argmax(start_logits)
            end_idx = np.argmax(end_logits)
            if start_idx > end_idx:
                start_idx, end_idx = end_idx, start_idx
            
            # Decode answer
            input_ids = inputs["input_ids"][0].cpu().numpy()
            tokens = tokenizer.convert_ids_to_tokens(input_ids)
            if start_idx < len(tokens) and end_idx < len(tokens):
                answer_tokens = tokens[start_idx:end_idx+1]
                pred_answer = tokenizer.convert_tokens_to_string(answer_tokens)
            else:
                pred_answer = ""
        
        em, f1 = compute_em_f1(pred_answer, true_answer)
        em_scores.append(em)
        f1_scores.append(f1)
    
    return np.mean(em_scores), np.mean(f1_scores)

# ===================================================
# 9. OPTIMIZER AND SCHEDULER
# ===================================================
optimizer = AdamW(
    model.parameters(),
    lr=config.LEARNING_RATE,
    weight_decay=config.WEIGHT_DECAY
)

num_training_steps = len(train_loader) * config.EPOCHS
num_warmup_steps = int(num_training_steps * config.WARMUP_RATIO)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps
)

# Mixed precision scaler
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler() if config.USE_FP16 else None

# ===================================================
# 10. TRAINING LOOP
# ===================================================
train_losses = []
val_losses = []
val_em_scores = []
val_f1_scores = []

print("\nStarting training...")
for epoch in range(config.EPOCHS):
    # Training
    model.train()
    total_train_loss = 0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.EPOCHS} [Train]")
    
    for batch in progress_bar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        start_positions = batch["start_positions"].to(device)
        end_positions = batch["end_positions"].to(device)
        
        optimizer.zero_grad()
        
        if config.USE_FP16:
            with autocast():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    start_positions=start_positions,
                    end_positions=end_positions
                )
                loss = outputs.loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                start_positions=start_positions,
                end_positions=end_positions
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()
        
        lr_scheduler.step()
        total_train_loss += loss.item()
        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
    
    avg_train_loss = total_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    
    # Validation loss
    model.eval()
    total_val_loss = 0
    val_progress = tqdm(val_loader, desc=f"Epoch {epoch+1}/{config.EPOCHS} [Val Loss]")
    with torch.no_grad():
        for batch in val_progress:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            start_positions = batch["start_positions"].to(device)
            end_positions = batch["end_positions"].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                start_positions=start_positions,
                end_positions=end_positions
            )
            loss = outputs.loss
            total_val_loss += loss.item()
            val_progress.set_postfix({"loss": f"{loss.item():.4f}"})
    
    avg_val_loss = total_val_loss / len(val_loader)
    val_losses.append(avg_val_loss)
    
    # Evaluate EM and F1 on the original validation dataframe (without stride)
    em, f1 = evaluate_model(model, val_df, tokenizer, device)
    val_em_scores.append(em)
    val_f1_scores.append(f1)
    
    print(f"\nEpoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}, EM = {em:.4f}, F1 = {f1:.4f}\n")

# ===================================================
# 11. SAVE MODEL
# ===================================================
model_save_path = "./amharic_qa_bert_model_fixed"
model.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)
print(f"\nModel saved to {model_save_path}")

# ===================================================
# 12. PLOT RESULTS
# ===================================================
plt.figure(figsize=(14,5))

plt.subplot(1,2,1)
plt.plot(range(1, config.EPOCHS+1), train_losses, 'b-o', label='Train Loss')
plt.plot(range(1, config.EPOCHS+1), val_losses, 'r-o', label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1,2,2)
plt.plot(range(1, config.EPOCHS+1), val_em_scores, 'g-o', label='Exact Match (EM)')
plt.plot(range(1, config.EPOCHS+1), val_f1_scores, 'm-o', label='F1 Score')
plt.xlabel('Epoch')
plt.ylabel('Score')
plt.title('Validation EM and F1 Scores')
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim(0,1)

plt.tight_layout()
plt.show()

# ===================================================
# 13. PREDICTION FUNCTION (optional)
# ===================================================
def predict_answer(question, context, model, tokenizer, device, max_len=384):
    model.eval()
    inputs = tokenizer(question, context, max_length=max_len, truncation="only_second", return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        start_idx = torch.argmax(outputs.start_logits[0]).item()
        end_idx = torch.argmax(outputs.end_logits[0]).item()
        if start_idx > end_idx:
            start_idx, end_idx = end_idx, start_idx
        input_ids = inputs["input_ids"][0].cpu().numpy()
        answer_tokens = tokenizer.convert_ids_to_tokens(input_ids[start_idx:end_idx+1])
        answer = tokenizer.convert_tokens_to_string(answer_tokens)
    return answer

# Test prediction with a sample from the validation set
if len(val_df) > 0:
    sample = val_df.iloc[0]
    pred = predict_answer(sample['Question'], sample['context'], model, tokenizer, device)
    print("\nSample Prediction:")
    print(f"Question: {sample['Question']}")
    print(f"Context: {sample['context'][:200]}...")
    print(f"Predicted Answer: {pred}")
    print(f"True Answer: {sample['Answer']}")
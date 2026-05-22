# train_bilstm_attention_amharic_qa.py
import os
import re
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import gensim
from gensim.models import KeyedVectors
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
# -------------------------
# Config
# -------------------------
W2V_PATH = "/content/amharic_word2vec.bin"
CSV_PATH = "/content/drive/MyDrive/PHD folder/QA dataset/AmhQA25442.csv"

HIDDEN_SIZE = 256
NUM_LAYERS = 3          # as requested
BATCH_SIZE = 16
NUM_EPOCHS = 5         # as requested
LR = 2e-4
MAX_LEN = 512
MODEL_NAME = "BiLSTM3_Attn_W2V"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# -------------------------
# Utilities: exact match, f1
# -------------------------
def normalize_text(s):
    return " ".join(str(s).strip().split())

def exact_match(pred, gold):
    return 1 if normalize_text(pred) == normalize_text(gold) else 0

def f1_score_span(pred, gold):
    p_tokens = normalize_text(pred).split()
    g_tokens = normalize_text(gold).split()
    if len(p_tokens) == 0 or len(g_tokens) == 0:
        return 1.0 if p_tokens == g_tokens else 0.0
    common = 0
    from collections import Counter
    pc = Counter(p_tokens)
    gc = Counter(g_tokens)
    for t in pc:
        common += min(pc[t], gc.get(t, 0))
    if common == 0:
        return 0.0
    prec = common / len(p_tokens)
    rec = common / len(g_tokens)
    return 2 * prec * rec / (prec + rec)

# -------------------------
# Load Word2Vec keyed vectors
# -------------------------
print("Loading Word2Vec embeddings from", W2V_PATH)
w2v = KeyedVectors.load_word2vec_format(W2V_PATH, binary=True)
EMBEDDING_DIM = w2v.vector_size  # use the actual dim; not 300
print("Loaded W2V dim:", EMBEDDING_DIM)

# -------------------------
# Tokenization simple whitespace + mapping
# -------------------------
def simple_tokenize(text):
    # basic whitespace tokenizer for Amharic; replace with better tokenizer if available
    return text.strip().split()

# -------------------------
# Build vocab from W2V and dataset
# -------------------------
print("Loading dataset", CSV_PATH)
df = pd.read_csv(CSV_PATH)[:1000]
# your CSV has: context, Question, Answer
assert {"context", "Question", "Answer"}.issubset(set(df.columns)), "CSV must contain context, Question, Answer columns"

UNK = "<UNK>"
PAD = "<PAD>"
vocab = {PAD: 0, UNK: 1}
idx = 2
for w in w2v.index_to_key:
    if w not in vocab:
        vocab[w] = idx
        idx += 1

def token_to_idx(tok):
    return vocab.get(tok, vocab[UNK])

# -------------------------
# Convert dataset to examples with span indices
# -------------------------
examples = []
missing_count = 0
for _, row in df.iterrows():
    context = str(row["context"])
    question = str(row["Question"])
    answer = str(row["Answer"])
    norm_context = context
    norm_answer = answer.strip()
    start_char = norm_context.find(norm_answer)
    if start_char == -1:
        missing_count += 1
        continue
    end_char = start_char + len(norm_answer)
    examples.append({"context": context, "Question": question, "Answer": answer,
                     "answer_start": start_char, "answer_end": end_char})
print(f"Total examples loaded: {len(examples)}  (skipped {missing_count} examples where answer not in context)")

# split
train_exs, temp = train_test_split(examples, test_size=0.2, random_state=SEED)
val_exs, test_exs = train_test_split(temp, test_size=0.5, random_state=SEED)
print("Splits: train", len(train_exs), "val", len(val_exs), "test", len(test_exs))

# -------------------------
# char_to_token_span
# -------------------------
def char_to_token_span(context, tokens, answer_start_char, answer_end_char):
    offsets = []
    cursor = 0
    for t in tokens:
        find_idx = context.find(t, cursor)
        if find_idx == -1:
            find_idx = cursor
        offsets.append((find_idx, find_idx + len(t)))
        cursor = offsets[-1][1]
    s_idx = None
    e_idx = None
    for i, (s, e) in enumerate(offsets):
        if s <= answer_start_char < e:
            s_idx = i
        if s < answer_end_char <= e:
            e_idx = i
    if s_idx is None:
        s_idx = 0
    if e_idx is None:
        e_idx = len(tokens) - 1
    return s_idx, e_idx

# -------------------------
# Dataset class
# -------------------------
class AmharicQADataset(Dataset):
    def __init__(self, exs, vocab, max_len=MAX_LEN):
        self.exs = exs
        self.vocab = vocab
        self.max_len = max_len
    def __len__(self): return len(self.exs)
    def __getitem__(self, idx):
        ex = self.exs[idx]
        context = ex["context"]
        q = ex["Question"]
        a = ex["Answer"]
        # For simplicity build single input: [question] + [SEP] + context
        q_tokens = simple_tokenize(q)
        c_tokens = simple_tokenize(context)
        s_token, e_token = char_to_token_span(context, c_tokens, ex["answer_start"], ex["answer_end"])
        sep = ["[SEP]"]
        tokens = q_tokens + sep + c_tokens
        context_token_start = len(q_tokens) + 1

        ids = [token_to_idx(t) for t in tokens]
        if len(ids) > self.max_len:
            ids = ids[:self.max_len]
        attn = [1] * len(ids)

        pad_len = self.max_len - len(ids)
        ids = ids + [vocab[PAD]] * pad_len
        attn = attn + [0] * pad_len

        start = context_token_start + s_token
        end = context_token_start + e_token
        start = min(start, self.max_len - 1)
        end = min(end, self.max_len - 1)

        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),           # shape (MAX_LEN,)
            "attention_mask": torch.tensor(attn, dtype=torch.long),     # shape (MAX_LEN,)
            "start_pos": torch.tensor(start, dtype=torch.long),         # scalar
            "end_pos": torch.tensor(end, dtype=torch.long),             # scalar
            "raw_context": context,                                     # str
            "raw_answer": a,                                            # str
            "tokens": tokens,                                           # List[str]
            "Question": q,                                              # str
        }

train_ds = AmharicQADataset(train_exs, vocab)
val_ds = AmharicQADataset(val_exs, vocab)
test_ds = AmharicQADataset(test_exs, vocab)

# -------------------------
# Fix: custom collate_fn that only stacks tensors
# -------------------------
def collate_fn(batch):
    # Only stack tensor fields; keep non‑tensor fields as plain lists.
    # PyTorch’s default collate_fn fails here because of variable‑length raw strings.
    tensors = ["input_ids", "attention_mask", "start_pos", "end_pos"]
    stacked = {}
    for k in tensors:
        stacked[k] = torch.stack([item[k] for item in batch])
    others = {}
    non_tensor_keys = ["raw_context", "raw_answer", "tokens", "Question"]
    for k in non_tensor_keys:
        others[k + "s"] = [item[k] for item in batch]  # list of str / List[str]
    stacked.update(others)
    return stacked

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, collate_fn=collate_fn)
test_loader = DataLoader(test_ds, batch_size=1, collate_fn=collate_fn)  # 1 to keep __getitem__ format easy

# -------------------------
# Prepare embedding matrix from w2v
# -------------------------
vocab_size = len(vocab)
embedding_matrix = np.random.normal(scale=0.6, size=(vocab_size, EMBEDDING_DIM)).astype(np.float32)
for w, i in vocab.items():
    if w in w2v:
        embedding_matrix[i] = w2v[w]
embedding_matrix[vocab[PAD]] = 0.0  # PAD is zeros

# -------------------------
# Model: BiLSTM (3 layers) + Attention -> span start/end logits
# -------------------------
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)
    def forward(self, H, mask=None):
        # H: (B, T, D)
        score = torch.tanh(self.linear(H))
        logits = self.v(score).squeeze(-1)
        if mask is not None:
            logits = logits.masked_fill(mask==0, -1e9)
        weights = torch.softmax(logits, dim=-1)
        context = torch.bmm(weights.unsqueeze(1), H).squeeze(1)
        return context, weights

class BiLSTM_Attn_Span(nn.Module):
    def __init__(self, embedding_matrix, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, dropout=0.2):
        super().__init__()
        emb_weight = torch.tensor(embedding_matrix)
        num_embeddings, emb_dim = emb_weight.size()
        self.embedding = nn.Embedding(num_embeddings, emb_dim, padding_idx=0)
        self.embedding.weight = nn.Parameter(emb_weight)
        self.embedding.weight.requires_grad = False  # freeze initially
        self.bilstm = nn.LSTM(input_size=emb_dim, hidden_size=hidden_size, num_layers=num_layers,
                              bidirectional=True, batch_first=True, dropout=dropout)
        self.attn = Attention(hidden_size*2)
        # span prediction heads
        self.start_proj = nn.Linear(hidden_size*2, 1)
        self.end_proj = nn.Linear(hidden_size*2, 1)
    def forward(self, input_ids, attention_mask):
        emb = self.embedding(input_ids)
        outputs, _ = self.bilstm(emb)
        # logits per token
        start_logits = self.start_proj(outputs).squeeze(-1)
        end_logits = self.end_proj(outputs).squeeze(-1)
        # mask
        start_logits = start_logits.masked_fill(attention_mask==0, -1e9)
        end_logits = end_logits.masked_fill(attention_mask==0, -1e9)
        return start_logits, end_logits, outputs

model = BiLSTM_Attn_Span(embedding_matrix).to(DEVICE)
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
loss_fn = nn.CrossEntropyLoss()

# -------------------------
# Training + evaluation helpers
# -------------------------
def compute_loss(start_logits, end_logits, start_pos, end_pos):
    loss_start = loss_fn(start_logits, start_pos)
    loss_end = loss_fn(end_logits, end_pos)
    return (loss_start + loss_end) / 2.0

def train_one_epoch(model, loader, optimizer):
    model.train()
    total_loss = 0.0
    pbar = tqdm(loader, desc="train", leave=False)
    for batch in pbar:
        # Unpack batch (tensors are already stacked; strings are in *_s lists)
        input_ids = batch["input_ids"].to(DEVICE)
        attn_mask = batch["attention_mask"].to(DEVICE)
        s_pos = batch["start_pos"].to(DEVICE)
        e_pos = batch["end_pos"].to(DEVICE)

        optimizer.zero_grad()
        s_logits, e_logits, _ = model(input_ids, attn_mask)
        loss = compute_loss(s_logits, e_logits, s_pos, e_pos)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def eval_loss(model, loader):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attn_mask = batch["attention_mask"].to(DEVICE)
            s_pos = batch["start_pos"].to(DEVICE)
            e_pos = batch["end_pos"].to(DEVICE)
            s_logits, e_logits, _ = model(input_ids, attn_mask)
            loss = compute_loss(s_logits, e_logits, s_pos, e_pos)
            total_loss += loss.item()
    return total_loss / len(loader)

def predict_span_text(model, batch_item):
    model.eval()
    with torch.no_grad():
        # batch_item is a dict of single‑example tensors
        input_ids = batch_item["input_ids"].unsqueeze(0).to(DEVICE)
        attn_mask = batch_item["attention_mask"].unsqueeze(0).to(DEVICE)
        s_logits, e_logits, outputs = model(input_ids, attn_mask)
        s_probs = torch.softmax(s_logits, dim=-1).cpu().numpy()[0]
        e_probs = torch.softmax(e_logits, dim=-1).cpu().numpy()[0]
        L = len(s_probs)
        best_score = -1e9
        best_s, best_e = 0, 0
        for si in range(L):
            for ei in range(si, min(si+30, L)):
                score = s_probs[si] * e_probs[ei]
                if score > best_score:
                    best_score = score
                    best_s, best_e = si, ei
        # Reconstruct tokens → text
        tokens = batch_item["tokens"]  # list of str
        best_s = min(best_s, len(tokens)-1)
        best_e = min(best_e, len(tokens)-1)
        pred_tokens = tokens[best_s:best_e+1]
        pred_text = " ".join(pred_tokens)
        return pred_text

# -------------------------
# Training loop
# -------------------------
train_losses = []
val_losses = []
for epoch in range(NUM_EPOCHS):
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
    train_loss = train_one_epoch(model, train_loader, optimizer)
    val_loss = eval_loss(model, val_loader)
    print(f" train_loss: {train_loss:.4f}  val_loss: {val_loss:.4f}")
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    # Unfreeze embeddings after 2 epochs (optional)
    if epoch == 1:
        model.embedding.weight.requires_grad = True

# Plot loss
plt.figure(figsize=(6,4))
plt.plot(range(1, NUM_EPOCHS+1), train_losses, label="train_loss")
plt.plot(range(1, NUM_EPOCHS+1), val_losses, label="val_loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and validation loss")
plt.legend()
plt.tight_layout()
plt.savefig("training_loss.png")
print("Saved training_loss.png")

# -------------------------
# Evaluation on test set: EM and F1
# -------------------------
all_em = []
all_f1 = []
pred_rows = []

for batch in tqdm(test_loader, desc="Evaluating"):
    # batch contains lists of 1 item because batch_size=1
    item = {
        k[:-1] if k.endswith("s") else k: v[0] for k, v in batch.items()
        if k not in ["input_ids", "attention_mask", "start_pos", "end_pos"]
    }
    preds = predict_span_text(model, batch)
    # batch["raw_answers"] is a list of 1
    gold = batch["raw_answers"][0]
    em = exact_match(preds, gold)
    f1 = f1_score_span(preds, gold)
    all_em.append(em)
    all_f1.append(f1)
    pred_rows.append({
        "Question": " ".join(item["tokens"][:min(30, len(item["tokens"]))]),
        "Gold_Answer": gold,
        "Predicted_Answer": preds,
        "EM": em,
        "F1": f1,
    })

em_avg = np.mean(all_em)
f1_avg = np.mean(all_f1)
print(f"Test EM: {em_avg:.4f}  Test F1: {f1_avg:.4f}")

# Save predictions
pred_df = pd.DataFrame(pred_rows)
pred_df.to_csv("amqa_predictions.csv", index=False)
print("Saved amqa_predictions.csv")

# Results table (Model, EM, F1)
results_table = pd.DataFrame([{
    "Model": MODEL_NAME,
    "EM": round(float(em_avg), 4),
    "F1": round(float(f1_avg), 4)
}])
results_table.to_csv("amqa_results.csv", index=False)
print(results_table.to_string(index=False))
#Dataset=rasyosef/amharic-passage-retrieval-dataset
#df = pd.read_csv("/content/drive/MyDrive/PHD folder/QA dataset/AmhQA2544.csv")  # Dataset columns: 'question', 'answer', 'context'
# ============================================================
# LSTM + ATTENTION EXTRACTIVE QA FOR AMHARIC (FULL CODE)
# ============================================================

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from gensim.models import KeyedVectors
import re
from collections import defaultdict
import matplotlib.pyplot as plt
from tqdm import tqdm

# -------------------------------
# 1. CONFIGURATION
# -------------------------------
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MAX_CONTEXT_LEN = 200
MAX_QUESTION_LEN = 30
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-3
HIDDEN_SIZE = 128
EMBEDDING_DIM = None  # will be set after loading word2vec

# -------------------------------
# 2. LOAD DATA
# -------------------------------
df = pd.read_csv("/content/drive/MyDrive/PHD folder/QA dataset/AmhQA25442.csv")
df = df.fillna('')
for col in ['context', 'Question', 'Answer']:
    if col in df.columns:
        df[col] = df[col].astype(str)

# -------------------------------
# 3. LOAD WORD2VEC EMBEDDINGS
# -------------------------------
word2vec = KeyedVectors.load_word2vec_format("/content/amharic_word2vec.bin", binary=True)
EMBEDDING_DIM = word2vec.vector_size
print(f"Embedding dimension: {EMBEDDING_DIM}")

# -------------------------------
# 4. TOKENIZATION & VOCABULARY
# -------------------------------
def tokenize_amharic(text):
    if not isinstance(text, str):
        return []
    # Keep only Amharic characters and spaces
    text = re.sub(r'[^\u1200-\u137F\s]', '', text)
    return text.strip().split()

# Build vocabulary from dataset + word2vec
word_vocab = defaultdict(lambda: len(word_vocab))
word_vocab['<PAD>'] = 0
word_vocab['<UNK>'] = 1

for _, row in df.iterrows():
    for col in ['context', 'Question', 'Answer']:
        for w in tokenize_amharic(row[col]):
            if w not in word_vocab and w in word2vec:
                word_vocab[w]

vocab_size = len(word_vocab)
print(f"Vocabulary size: {vocab_size}")

# Embedding matrix
embedding_matrix = np.zeros((vocab_size, EMBEDDING_DIM))
for word, idx in word_vocab.items():
    if word in word2vec:
        embedding_matrix[idx] = word2vec[word]
    else:
        embedding_matrix[idx] = np.random.normal(scale=0.1, size=EMBEDDING_DIM)

# Encoding function (fixed length)
def encode_text(text, max_len):
    tokens = tokenize_amharic(text)[:max_len]
    ids = [word_vocab.get(t, word_vocab['<UNK>']) for t in tokens]
    ids += [word_vocab['<PAD>']] * (max_len - len(ids))
    return np.array(ids, dtype=np.int64)

# -------------------------------
# 5. EXTRACT ANSWER SPANS
# -------------------------------
def find_answer_span(context, answer):
    ctx_tokens = tokenize_amharic(context)
    ans_tokens = tokenize_amharic(answer)
    if not ans_tokens:
        return 0, 0
    for i in range(len(ctx_tokens) - len(ans_tokens) + 1):
        if ctx_tokens[i:i+len(ans_tokens)] == ans_tokens:
            return i, i+len(ans_tokens)-1
    return 0, 0  # fallback

# Prepare data
contexts, questions, start_positions, end_positions, raw_contexts = [], [], [], [], []
for _, row in df.iterrows():
    ctx = row['context']
    q = row['Question']
    ans = row['Answer']
    if len(ctx) == 0 or len(q) == 0:
        continue
    s, e = find_answer_span(ctx, ans)
    if s >= MAX_CONTEXT_LEN or e >= MAX_CONTEXT_LEN:
        continue
    contexts.append(encode_text(ctx, MAX_CONTEXT_LEN))
    questions.append(encode_text(q, MAX_QUESTION_LEN))
    start_positions.append(s)
    end_positions.append(e)
    raw_contexts.append(ctx)

contexts = torch.tensor(contexts)
questions = torch.tensor(questions)
start_positions = torch.tensor(start_positions)
end_positions = torch.tensor(end_positions)
print(f"Total valid examples: {len(contexts)}")

# -------------------------------
# 6. DATASET & DATALOADER
# -------------------------------
class QADataset(Dataset):
    def __init__(self, ctx, q, s, e, raw_ctx):
        self.ctx = ctx
        self.q = q
        self.s = s
        self.e = e
        self.raw_ctx = raw_ctx
    def __len__(self):
        return len(self.ctx)
    def __getitem__(self, idx):
        return self.ctx[idx], self.q[idx], self.s[idx], self.e[idx], self.raw_ctx[idx]

# Train/val split
n_total = len(contexts)
n_train = int(0.8 * n_total)
indices = torch.randperm(n_total)
train_idx, val_idx = indices[:n_train], indices[n_train:]

train_raw = [raw_contexts[i] for i in train_idx.tolist()]
val_raw   = [raw_contexts[i] for i in val_idx.tolist()]

train_dataset = QADataset(contexts[train_idx], questions[train_idx],
                          start_positions[train_idx], end_positions[train_idx], train_raw)
val_dataset   = QADataset(contexts[val_idx], questions[val_idx],
                          start_positions[val_idx], end_positions[val_idx], val_raw)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# -------------------------------
# 7. LSTM + ATTENTION MODEL
# -------------------------------
class AttentionQAModel(nn.Module):
    def __init__(self, embedding_matrix, hidden_size=128):
        super().__init__()
        vocab_size, emb_dim = embedding_matrix.shape
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))
        self.embedding.weight.requires_grad = False  # freeze embeddings

        self.lstm_context = nn.LSTM(emb_dim, hidden_size, batch_first=True, bidirectional=True)
        self.lstm_question = nn.LSTM(emb_dim, hidden_size, batch_first=True, bidirectional=True)
        self.output_layer = nn.Linear(hidden_size * 4, 2)  # concat ctx and attended_q

    def forward(self, context, question):
        # context: (batch, ctx_len), question: (batch, q_len)
        ctx_emb = self.embedding(context)   # (batch, ctx_len, emb_dim)
        q_emb = self.embedding(question)    # (batch, q_len, emb_dim)

        ctx_out, _ = self.lstm_context(ctx_emb)   # (batch, ctx_len, 2*hidden)
        q_out, _ = self.lstm_question(q_emb)      # (batch, q_len, 2*hidden)

        # Attention: context attends to question
        attn_weights = torch.matmul(ctx_out, q_out.transpose(1, 2))  # (batch, ctx_len, q_len)
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attended_q = torch.matmul(attn_weights, q_out)               # (batch, ctx_len, 2*hidden)

        combined = torch.cat([ctx_out, attended_q], dim=-1)          # (batch, ctx_len, 4*hidden)
        logits = self.output_layer(combined)                         # (batch, ctx_len, 2)
        start_logits = logits[:, :, 0]
        end_logits   = logits[:, :, 1]
        return start_logits, end_logits

# -------------------------------
# 8. METRICS
# -------------------------------
def compute_em_f1(pred_start, pred_end, true_start, true_end, context_text):
    ctx_tokens = tokenize_amharic(context_text)
    pred_tokens = ctx_tokens[pred_start:pred_end+1] if pred_start <= pred_end else []
    true_tokens = ctx_tokens[true_start:true_end+1] if true_start <= true_end else []

    em = int(pred_tokens == true_tokens)

    if not pred_tokens and not true_tokens:
        f1 = 1.0
    elif not pred_tokens or not true_tokens:
        f1 = 0.0
    else:
        common = len(set(pred_tokens) & set(true_tokens))
        prec = common / len(pred_tokens)
        rec = common / len(true_tokens)
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return em, f1

def evaluate(model, loader):
    model.eval()
    total_em, total_f1 = 0, 0
    with torch.no_grad():
        for batch_ctx, batch_q, batch_s, batch_e, batch_raw in loader:
            batch_ctx = batch_ctx.to(DEVICE)
            batch_q   = batch_q.to(DEVICE)
            start_logits, end_logits = model(batch_ctx, batch_q)
            pred_start = torch.argmax(start_logits, dim=1).cpu().numpy()
            pred_end   = torch.argmax(end_logits, dim=1).cpu().numpy()
            true_start = batch_s.numpy()
            true_end   = batch_e.numpy()
            for i in range(len(batch_ctx)):
                em, f1 = compute_em_f1(pred_start[i], pred_end[i], true_start[i], true_end[i], batch_raw[i])
                total_em += em
                total_f1 += f1
    return total_em / len(loader.dataset), total_f1 / len(loader.dataset)

# -------------------------------
# 9. TRAINING LOOP
# -------------------------------
model = AttentionQAModel(embedding_matrix, hidden_size=HIDDEN_SIZE).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

train_losses, val_losses = [], []
val_em_scores, val_f1_scores = [], []

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    loop = tqdm(train_loader, desc=f'Epoch {epoch+1}/{EPOCHS}')

    for batch_ctx, batch_q, batch_s, batch_e, _ in loop:
        batch_ctx = batch_ctx.to(DEVICE)
        batch_q   = batch_q.to(DEVICE)
        batch_s   = batch_s.to(DEVICE)
        batch_e   = batch_e.to(DEVICE)

        optimizer.zero_grad()
        start_logits, end_logits = model(batch_ctx, batch_q)
        loss = criterion(start_logits, batch_s) + criterion(end_logits, batch_e)
        loss.backward()
        # Gradient clipping to avoid explosion
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    avg_train_loss = total_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # Validation loss
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch_ctx, batch_q, batch_s, batch_e, _ in val_loader:
            batch_ctx = batch_ctx.to(DEVICE)
            batch_q   = batch_q.to(DEVICE)
            batch_s   = batch_s.to(DEVICE)
            batch_e   = batch_e.to(DEVICE)
            start_logits, end_logits = model(batch_ctx, batch_q)
            loss = criterion(start_logits, batch_s) + criterion(end_logits, batch_e)
            val_loss += loss.item()
    avg_val_loss = val_loss / len(val_loader)
    val_losses.append(avg_val_loss)

    # Compute EM & F1 on validation set
    em, f1 = evaluate(model, val_loader)
    val_em_scores.append(em)
    val_f1_scores.append(f1)

    print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}, EM = {em:.4f}, F1 = {f1:.4f}")

# -------------------------------
# 10. PLOTTING
# -------------------------------
plt.figure(figsize=(12,4))

plt.subplot(1,2,1)
plt.plot(range(1, EPOCHS+1), train_losses, label='Train Loss')
plt.plot(range(1, EPOCHS+1), val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)

plt.subplot(1,2,2)
plt.plot(range(1, EPOCHS+1), val_em_scores, label='EM')
plt.plot(range(1, EPOCHS+1), val_f1_scores, label='F1')
plt.xlabel('Epoch')
plt.ylabel('Score')
plt.title('Validation EM and F1')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# -------------------------------
# 11. EXAMPLE PREDICTION
# -------------------------------
def predict_answer(question, context, model, max_ctx=MAX_CONTEXT_LEN, max_q=MAX_QUESTION_LEN):
    model.eval()
    ctx_enc = torch.tensor([encode_text(context, max_ctx)]).to(DEVICE)
    q_enc   = torch.tensor([encode_text(question, max_q)]).to(DEVICE)
    with torch.no_grad():
        start_logits, end_logits = model(ctx_enc, q_enc)
        start_idx = torch.argmax(start_logits[0]).item()
        end_idx   = torch.argmax(end_logits[0]).item()
        if start_idx > end_idx:
            start_idx, end_idx = end_idx, start_idx
        ctx_tokens = tokenize_amharic(context)
        if start_idx < len(ctx_tokens) and end_idx < len(ctx_tokens):
            return ' '.join(ctx_tokens[start_idx:end_idx+1])
        else:
            return ""

# Test with first validation example
if len(val_dataset) > 0:
    sample_ctx = val_raw[0]
    sample_q = df.iloc[val_idx[0].item()]['Question']
    pred = predict_answer(sample_q, sample_ctx, model)
    print("\nSample Prediction:")
    print(f"Question: {sample_q}")
    print(f"Context: {sample_ctx[:150]}...")
    print(f"Predicted Answer: {pred}")
    true_ans = df.iloc[val_idx[0].item()]['Answer']
    print(f"True Answer: {true_ans}")
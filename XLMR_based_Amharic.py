#XLM-R Based QA
# ==========================================================
# Amharic Question Answering using XLM-R
# Train-Test Split Evaluation
# ==========================================================

import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt

from collections import Counter
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    TrainingArguments,
    Trainer
)

# ==========================================================
# Load Dataset
# ==========================================================

df = pd.read_csv(
    "/content/drive/MyDrive/PHD folder/QA dataset/AmhQA2544.csv"
)

print("Dataset Size:", len(df))
print(df.columns)

# ==========================================================
# Find Answer Start Positions
# ==========================================================

def find_answer_start(context, answer):
    return context.find(answer)

df["answer_start"] = df.apply(
    lambda x: find_answer_start(
        str(x["context"]),
        str(x["Answer"])
    ),
    axis=1
)

# Remove rows where answer not found
df = df[df["answer_start"] != -1].reset_index(drop=True)

print("Valid Samples:", len(df))

# ==========================================================
# EM and F1 Functions
# ==========================================================

def normalize_answer(s):
    return str(s).strip().lower()

def compute_em(pred, gold):
    return int(normalize_answer(pred) == normalize_answer(gold))

def compute_f1(pred, gold):

    pred_tokens = normalize_answer(pred).split()
    gold_tokens = normalize_answer(gold).split()

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)

    return 2 * precision * recall / (precision + recall)

# ==========================================================
# Train-Test Split
# ==========================================================

train_df, test_df = train_test_split(
    df,
    test_size=0.20,
    random_state=42,
    shuffle=True
)

print("Train Size:", len(train_df))
print("Test Size :", len(test_df))

# ==========================================================
# Dataset Class
# ==========================================================

class QADataset(Dataset):

    def __init__(
        self,
        dataframe,
        tokenizer,
        max_length=384
    ):
        self.data = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        question = str(
            self.data.loc[idx, "Question"]
        )

        context = str(
            self.data.loc[idx, "context"]
        )

        answer = str(
            self.data.loc[idx, "Answer"]
        )

        answer_start = int(
            self.data.loc[idx, "answer_start"]
        )

        answer_end = answer_start + len(answer)

        encoding = self.tokenizer(
            question,
            context,
            max_length=self.max_length,
            truncation="only_second",
            padding="max_length",
            return_offsets_mapping=True,
            return_tensors="pt"
        )

        offsets = encoding.pop(
            "offset_mapping"
        )[0]

        start_token = 0
        end_token = 0

        for i, (start, end) in enumerate(offsets.tolist()):

            if start <= answer_start < end:
                start_token = i

            if start < answer_end <= end:
                end_token = i
                break

        return {
            "input_ids":
                encoding["input_ids"].squeeze(),

            "attention_mask":
                encoding["attention_mask"].squeeze(),

            "start_positions":
                torch.tensor(
                    start_token,
                    dtype=torch.long
                ),

            "end_positions":
                torch.tensor(
                    end_token,
                    dtype=torch.long
                )
        }

# ==========================================================
# Tokenizer and Datasets
# ==========================================================

tokenizer = AutoTokenizer.from_pretrained(
    "xlm-roberta-base"
)

train_dataset = QADataset(
    train_df,
    tokenizer
)

test_dataset = QADataset(
    test_df,
    tokenizer
)

# ==========================================================
# Load Model
# ==========================================================

model = AutoModelForQuestionAnswering.from_pretrained(
    "xlm-roberta-base"
)

# ==========================================================
# Training Arguments
# ==========================================================

training_args = TrainingArguments(
    output_dir="./xlmr_amharic_qa",

    num_train_epochs=3,

    per_device_train_batch_size=8,

    per_device_eval_batch_size=8,

    eval_strategy="epoch",

    save_strategy="epoch",

    logging_strategy="epoch",

    load_best_model_at_end=True,

    report_to="none"
)

# ==========================================================
# Trainer
# ==========================================================

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

# ==========================================================
# Train Model
# ==========================================================

trainer.train()

# ==========================================================
# Prediction
# ==========================================================

predictions = trainer.predict(
    test_dataset
)

start_logits, end_logits = (
    predictions.predictions
)

# ==========================================================
# Calculate EM and F1
# ==========================================================

em_scores = []
f1_scores = []

for i in range(len(test_dataset)):

    input_ids = test_dataset[i]["input_ids"]

    start_idx = np.argmax(
        start_logits[i]
    )

    end_idx = np.argmax(
        end_logits[i]
    )

    if start_idx > end_idx:
        end_idx = start_idx

    pred_tokens = input_ids[
        start_idx:end_idx+1
    ]

    pred_answer = tokenizer.decode(
        pred_tokens,
        skip_special_tokens=True
    )

    gold_answer = str(
        test_df.iloc[i]["Answer"]
    )

    em_scores.append(
        compute_em(
            pred_answer,
            gold_answer
        )
    )

    f1_scores.append(
        compute_f1(
            pred_answer,
            gold_answer
        )
    )

# ==========================================================
# Final Metrics
# ==========================================================

EM = np.mean(em_scores) * 100
F1 = np.mean(f1_scores) * 100
LOSS = predictions.metrics["test_loss"]

# ==========================================================
# Results Table
# ==========================================================

results_df = pd.DataFrame({
    "Metric": [
        "Exact Match (EM)",
        "F1 Score",
        "Loss"
    ],
    "Score": [
        round(EM, 2),
        round(F1, 2),
        round(LOSS, 4)
    ]
})

print("\n==============================")
print("Evaluation Results")
print("==============================")
print(results_df)

# ==========================================================
# Summary Table
# ==========================================================

summary_df = pd.DataFrame({
    "Train Samples": [len(train_df)],
    "Test Samples": [len(test_df)],
    "EM (%)": [round(EM, 2)],
    "F1 (%)": [round(F1, 2)],
    "Loss": [round(LOSS, 4)]
})

print("\n==============================")
print("Summary")
print("==============================")
print(summary_df)

# ==========================================================
# Performance Graph
# ==========================================================
plt.figure(figsize=(8,5))

metrics = ["EM", "F1", "Loss"]
values = [EM, F1, LOSS]

bars = plt.bar(metrics, values)

for bar in bars:
    height = bar.get_height()

    plt.text(
        bar.get_x() + bar.get_width()/2,
        height,
        f"{height:.2f}",
        ha="center"
    )

plt.title(
    "XLM-R Amharic QA Performance"
)

plt.ylabel("Score")
plt.grid(True, alpha=0.3)

plt.show()
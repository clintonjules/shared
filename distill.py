import os

# Must be set before importing transformers/datasets so neither library
# attempts any outbound call (model lookup, telemetry, etc.)
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["WANDB_DISABLED"] = "true"

import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)

# Local path to the already-downloaded model weights/tokenizer files
# (a directory containing config.json, tokenizer files, *.safetensors, etc.)
model_name = "/path/to/local/gemma-model"
csv_path = "data.csv"
max_length = 4096

tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

df = pd.read_csv(csv_path).fillna("")
df["prompt"] = df["system_prompt"].astype(str) + "\n\nUser: " + df["question"].astype(str) + "\n\nAssistant:"
df["target"] = df["response"].astype(str)


def tokenize(row):
    prompt_ids = tokenizer(row["prompt"], add_special_tokens=False)["input_ids"]
    full = tokenizer(row["prompt"] + " " + row["target"], truncation=True, max_length=max_length)
    labels = full["input_ids"].copy()
    cut = min(len(prompt_ids), len(labels))
    labels[:cut] = [-100] * cut
    full["labels"] = labels
    return full


dataset = Dataset.from_pandas(df[["prompt", "target"]]).map(
    tokenize, remove_columns=["prompt", "target"]
)

model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.bfloat16, device_map="auto", local_files_only=True
)

collator = DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100)

args = TrainingArguments(
    output_dir="gemma4-distilled",
    per_device_train_batch_size=1,
    num_train_epochs=1,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    logging_steps=20,
    bf16=True,
    save_strategy="epoch",
    report_to="none",  # no wandb/mlflow/tensorboard-hub callbacks, no external transmission
)

trainer = Trainer(model=model, args=args, train_dataset=dataset, data_collator=collator)
trainer.train()

trainer.save_model("gemma4-distilled")
tokenizer.save_pretrained("gemma4-distilled")

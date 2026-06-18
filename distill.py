import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup

csv_path = "data.csv"
model_name = "google/gemma-4-31B-it"

df = pd.read_csv(csv_path).fillna("")
df["prompt"] = df["system_prompt"].astype(str) + "\n\nUser: " + df["question"].astype(str) + "\n\nAssistant:"
df["target"] = df["response"].astype(str)

tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

class DistillCSV(Dataset):
    def __init__(self, frame, tok, max_length=4096):
        self.df = frame.reset_index(drop=True)
        self.tok = tok
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        prompt = row["prompt"]
        target = row["target"]

        prompt_ids = self.tok(prompt, add_special_tokens=False)["input_ids"]
        full = self.tok(
            prompt + " " + target,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        input_ids = full["input_ids"].squeeze(0)
        attention_mask = full["attention_mask"].squeeze(0)
        labels = input_ids.clone()
        cut = min(len(prompt_ids), len(labels))
        labels[:cut] = -100

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

def collate_fn(batch, pad_id):
    max_len = max(len(x["input_ids"]) for x in batch)

    input_ids, attention_mask, labels = [], [], []
    for x in batch:
        pad = max_len - len(x["input_ids"])
        input_ids.append(torch.cat([x["input_ids"], torch.full((pad,), pad_id, dtype=torch.long)]))
        attention_mask.append(torch.cat([x["attention_mask"], torch.zeros(pad, dtype=torch.long)]))
        labels.append(torch.cat([x["labels"], torch.full((pad,), -100, dtype=torch.long)]))

    return {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_mask),
        "labels": torch.stack(labels),
    }

ds = DistillCSV(df, tokenizer, max_length=4096)
loader = DataLoader(ds, batch_size=1, shuffle=True, collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id))

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
model.train()

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
epochs = 1
steps = epochs * len(loader)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=max(1, int(0.1 * steps)),
    num_training_steps=steps
)

for epoch in range(epochs):
    for step, batch in enumerate(loader):
        batch = {k: v.to(model.device) for k, v in batch.items()}
        out = model(**batch)
        loss = out.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        if step % 20 == 0:
            print(f"epoch={epoch} step={step} loss={loss.item():.4f}")

model.save_pretrained("gemma4-distilled")
tokenizer.save_pretrained("gemma4-distilled")

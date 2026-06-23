import pandas as pd
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, get_peft_model
import torch

# ========================= CONFIG =========================
csv_path = "your_data.csv"          # <-- Change this
student_model_name = "meta-llama/Llama-3.2-1B-Instruct"   # or 3B, 8B, etc.

# Column names in your CSV (change if different)
PROMPT_COL = "prompt"
QUESTION_COL = "question"
RESPONSE_COL = "oss_response"       # teacher response
# =========================================================

# Load CSV
df = pd.read_csv(csv_path)

# Convert to Hugging Face Dataset
dataset = Dataset.from_pandas(df)

# Formatting function - creates proper training text
def format_for_training(example):
    prompt = example[PROMPT_COL]
    question = example[QUESTION_COL]
    response = example[RESPONSE_COL]
    
    # Combine prompt + question
    user_message = f"{prompt}\n\n{question}".strip()
    
    # Llama-3 style chat format (recommended)
    text = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

{user_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{response}<|eot_id|>"""
    
    return {"text": text}

# Apply formatting
dataset = dataset.map(format_for_training, remove_columns=dataset.column_names)

print(f"Dataset loaded with {len(dataset)} examples")
print("\nSample example:\n", dataset[0]["text"][:500] + "...")

# ========================= MODEL & TOKENIZER =========================
tokenizer = AutoTokenizer.from_pretrained(student_model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    student_model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="flash_attention_2",   # remove if not supported
)

# LoRA (highly recommended)
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules="all-linear",
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ========================= TRAINING =========================
training_args = SFTConfig(
    output_dir="./llama_oss_distilled",
    per_device_train_batch_size=2,           # adjust based on GPU
    gradient_accumulation_steps=8,
    learning_rate=2e-5,
    num_train_epochs=2,
    bf16=True,
    logging_steps=10,
    save_strategy="epoch",
    optim="adamw_torch",
    max_seq_length=2048,
    packing=True,                            # very important for efficiency
    dataset_text_field="text",
    report_to="none",                        # change to "wandb" if you use it
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=training_args,
)

trainer.train()
trainer.save_model("./llama_oss_distilled_final")
tokenizer.save_pretrained("./llama_oss_distilled_final")

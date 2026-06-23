import torch
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer, SFTConfig

# Load your dataset (assume JSONL or CSV with columns: prompt/question, oss_response)
# Example: dataset = load_dataset("json", data_files="your_data.jsonl")

def format_example(example):
    # Customize this to your format
    prompt = example["prompt"] + "\n" + example["question"]
    response = example["oss_response"]
    # Use chat template if available, or simple format
    return {
        "text": f"<|user|>\n{prompt}<|assistant|>\n{response}"
        # or for Llama-3 style: apply_chat_template
    }

# Apply formatting
dataset = your_dataset.map(format_example)

model_name = "meta-llama/Llama-3.2-1B-Instruct"  # or your student (smaller Llama)
teacher_responses = ...  # your OSS data

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="flash_attention_2",  # if supported
)

# Optional: LoRA for efficiency
from peft import LoraConfig, get_peft_model
lora_config = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules="all-linear",  # or ["q_proj", "v_proj", ...]
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

training_args = TrainingArguments(
    output_dir="./llama_distilled",
    per_device_train_batch_size=4,  # adjust to your GPU
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    num_train_epochs=2,
    fp16=False,
    bf16=True,
    logging_steps=10,
    save_strategy="epoch",
    optim="adamw_torch",
    report_to="none",  # or "wandb"
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=SFTConfig(  # or TrainingArguments
        max_seq_length=2048,
        packing=True,  # efficient for multiple short examples
        dataset_text_field="text",
    ),
)

trainer.train()
trainer.save_model("./llama_distilled_final")

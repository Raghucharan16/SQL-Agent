import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import intel_extension_for_pytorch as ipex

# Load model and tokenizer
model_name = "defog/sqlcoder-7b"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_4bit=True,
    device_map="cpu",
    torch_dtype=torch.bfloat16
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Optimize for Intel CPU
model = ipex.optimize(model)

# Configure LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# Enable gradient checkpointing
model.gradient_checkpointing_enable()

# Load dataset
dataset = load_dataset("json", data_files="training_data.json")

# Tokenize dataset
def preprocess_function(examples):
    prompt = [f"### Schema:\n{s}\n\n### Question:\n{q}\n\n### SQL Query:\n{a}" 
              for s, q, a in zip(examples["schema"], examples["question"], examples["sql_query"])]
    return tokenizer(prompt, truncation=True, padding="max_length", max_length=512)

tokenized_dataset = dataset.map(preprocess_function, batched=True)

# Training arguments
training_args = TrainingArguments(
    output_dir="./sqlcoder_finetuned",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=False,  # Use bfloat16 on CPU
    save_strategy="epoch",
    logging_steps=10,
)

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
)

# Start fine-tuning
trainer.train()

# Save fine-tuned model
model.save_pretrained("./sqlcoder_finetuned")
tokenizer.save_pretrained("./sqlcoder_finetuned")

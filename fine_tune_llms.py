# Import modules
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split

# Choose a pre-trained model
model_name = "gpt2"

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Prepare custom dataset
dataset = [
    {"input_text": "Hi", "output_text": "Hello"},
    {"input_text": "How are you?", "output_text": "I'm good, thanks"},
    {"input_text": "What do you like to do?", "output_text": "I like to read books and watch movies"},
    # ...
]

# Preprocess dataset
def preprocess(example):
    # Encode input text
    input_encoding = tokenizer(example["input_text"], return_tensors="pt")
    # Encode output text with eos token
    output_encoding = tokenizer(example["output_text"] + tokenizer.eos_token, return_tensors="pt")
    # Return input ids, attention mask, labels (same as output ids)
    return {
        "input_ids": input_encoding["input_ids"],
        "attention_mask": input_encoding["attention_mask"],
        "labels": output_encoding["input_ids"],
    }

dataset = [preprocess(example) for example in dataset]

# Split dataset into train and validation sets
train_dataset, val_dataset = train_test_split(dataset, test_size=0.2)

# Define training arguments
training_args = TrainingArguments(
    output_dir="output",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=2e-5,
    evaluation_strategy="epoch",
)

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Train model
trainer.train()

# Evaluate model
trainer.evaluate()

# Save model and tokenizer
model.save_pretrained("output")
tokenizer.save_pretrained("output")

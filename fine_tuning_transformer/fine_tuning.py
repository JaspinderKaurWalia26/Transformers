import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments
from transformers import Trainer
import transformers
print(transformers.__version__)

# loading dataset
dataset = load_dataset("ag_news")
print(dataset)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# tokenization function
def tokenize_function(example):
    return tokenizer(
        example["text"],
        padding="max_length",
        truncation=True
    )
    
# applying mapping
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# data formatting for pytorch
tokenized_dataset.set_format("torch")

# defining model

model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=4
)




# training arguments 
training_args = TrainingArguments(
    output_dir="results",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
)


# compute metrics function
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, average="weighted")
    }


# trainer api
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    compute_metrics=compute_metrics,
)
# starting training
trainer.train()

# evaluating
trainer.evaluate()

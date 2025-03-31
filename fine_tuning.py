from transformers import AutoModelForMaskedLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling, \
    BertTokenizer
from datasets import load_dataset
import gdown

TOKENIZER_PATH = "https://drive.google.com/drive/folders/1wwHMSLUsdf5ZwHPxXLVY8F0VaXfGD2Q_"
DATASET_DRIVE_LINK = "https://drive.google.com/uc?id=11tSWlA_TU0eESvAxo31-Ir5yi3LPbVLy"
MODEL_NAME = "tbs17/MathBERT-custom"

bert_cased_tokenizer = BertTokenizer.from_pretrained(
    pretrained_model_name_or_path=TOKENIZER_PATH,
    do_lower_case=False,
    truncation=True,
    model_max_length=512
)

model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME)
model.resize_token_embeddings(len(bert_cased_tokenizer))

gdown.download(DATASET_DRIVE_LINK, output="./dataset.txt", quiet=False)
dataset = load_dataset("text", data_files={"train": "./dataset.txt"})

tokenized_datasets = dataset.map(
    lambda examples: bert_cased_tokenizer(
        examples["text"],
        truncation=True,
        max_length=512,
        return_special_tokens_mask=True
    ),
    batched=True,
    remove_columns=["text"]
)

split_dataset = tokenized_datasets["train"].train_test_split(test_size=0.05)
train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="steps",
    eval_steps=3000,
    save_strategy="steps",
    save_steps=3000,
    save_total_limit=3,
    learning_rate=3e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=1000,
    warmup_ratio=0.06,
    report_to="none",
    fp16=True,
    dataloader_num_workers=4,
    gradient_accumulation_steps=2,
    push_to_hub=False,
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=bert_cased_tokenizer,
    mlm=True,
    mlm_probability=0.15
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
)

trainer.train()
trainer.save_model("./fine_tuned_mathbert")

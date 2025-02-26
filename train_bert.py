from transformers import BertTokenizer
from transformers import BertConfig, BertForPreTraining
from transformers import TextDatasetForNextSentencePrediction
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments


TOKENIZER_PATH = ""
DATASET_PATH = ""

bert_cased_tokenizer = BertTokenizer.from_pretrained(
    pretrained_model_name_or_path=TOKENIZER_PATH,
    do_lower_case=False,
    truncation=True,
    model_max_length=512
)

config = BertConfig(max_position_embeddings=512)
model = BertForPreTraining(config)
model.resize_token_embeddings(len(bert_cased_tokenizer))

dataset = TextDatasetForNextSentencePrediction(
    tokenizer=bert_cased_tokenizer,
    file_path=DATASET_PATH,
    block_size=512
)

for i in range(len(dataset)):
    example = dataset[i]

    if len(example['input_ids']) > 512:
        example['input_ids'] = example['input_ids'][:512]

        if 'attention_mask' in example:
            example['attention_mask'] = example['attention_mask'][:512]

        if 'token_type_ids' in example:
            example['token_type_ids'] = example['token_type_ids'][:512]

data_collator = DataCollatorForLanguageModeling(
    tokenizer=bert_cased_tokenizer,
    mlm=True,
    mlm_probability=0.15,
    return_tensors="pt",
    pad_to_multiple_of=512
)

training_args = TrainingArguments(
    output_dir="./output/",
    overwrite_output_dir=True,
    num_train_epochs=2,
    per_device_train_batch_size=2,
    save_steps=10_000,
    save_total_limit=2,
    prediction_loss_only=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

trainer.train()
trainer.save_model("./output/model/")

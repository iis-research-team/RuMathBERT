import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertTokenizerFast, BertConfig, BertForMaskedLM, AdamW
from tqdm import tqdm

df = pd.read_csv("filtered_formulas.csv", encoding="cp1251")

df['combined_text'] = df.apply(
    lambda row: f"{row['left_context']} [FORMULA] {row['formula']} [FORMULA] {row['right_context']}", axis=1)

tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
special_tokens = {"additional_special_tokens": ["[FORMULA]"]}
tokenizer.add_special_tokens(special_tokens)

batch_size = 128
all_input_ids = []
all_attention_masks = []

for i in tqdm(range(0, len(df), batch_size)):
    batch_texts = df['combined_text'][i:i + batch_size].tolist()

    encoded_batch = tokenizer(batch_texts,
                              padding=True,
                              truncation=True,
                              max_length=512,
                              return_tensors="pt")

    all_input_ids.append(encoded_batch["input_ids"])
    all_attention_masks.append(encoded_batch["attention_mask"])

max_length = max([tensor.shape[1] for tensor in all_input_ids])

padded_input_ids = [torch.nn.functional.pad(tensor, (0, max_length - tensor.shape[1]), value=tokenizer.pad_token_id)
                    for tensor in all_input_ids]

padded_attention_masks = [torch.nn.functional.pad(tensor, (0, max_length - tensor.shape[1]), value=0)
                          for tensor in all_attention_masks]

input_ids = torch.cat(padded_input_ids)
attention_masks = torch.cat(padded_attention_masks)

config = BertConfig(
    vocab_size=tokenizer.vocab_size,
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    intermediate_size=3072
)

model = BertForMaskedLM(config)
model.resize_token_embeddings(len(tokenizer))

optimizer = AdamW(model.parameters(), lr=5e-5)

train_dataset = TensorDataset(input_ids, attention_masks)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

model.train()
epochs = 3

total_steps = epochs * len(train_loader)
progress_bar = tqdm(total=total_steps, desc="Training: ")

for epoch in range(epochs):
    total_loss = 0
    for batch in train_loader:
        input_ids_batch, attention_masks_batch = batch
        input_ids_batch = input_ids_batch.to(device)
        attention_masks_batch = attention_masks_batch.to(device)

        outputs = model(input_ids=input_ids_batch, attention_mask=attention_masks_batch, labels=input_ids_batch)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress_bar.update(1)

    print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader)}")

progress_bar.close()

model.save_pretrained("./bert")
tokenizer.save_pretrained("./bert")

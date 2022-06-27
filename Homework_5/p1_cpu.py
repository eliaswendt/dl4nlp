import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
from datasets import load_dataset
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
from matplotlib import pyplot as plt

# ------------------------------------------------
#            1.2 DATA AND MODEL PREPARATION
# ------------------------------------------------
torch.manual_seed(0) # set manual seed for reproducibility
dataset = load_dataset("rotten_tomatoes")
model = AutoModelForSequenceClassification.from_pretrained("microsoft/xtremedistil-l6-h256-uncased", num_labels=2)
tokenizer = AutoTokenizer.from_pretrained("microsoft/xtremedistil-l6-h256-uncased")

# ------------------------------------------------
#            1.3 PREPROCESSING
# ------------------------------------------------
def tokenization(file):
    return tokenizer(file["text"], padding="max_length", max_length=256, truncation=True)

tokenized_dataset = dataset.map(tokenization, batched=True) # use map function to quickly tokenize batches of examples

tokenized_dataset = tokenized_dataset.remove_columns(["text"])
tokenized_dataset = tokenized_dataset.rename_column("label", "labels")

tokenized_dataset.set_format("torch")

dataset_train = tokenized_dataset["train"].shuffle(seed=123).select(range(2000))
dataset_validation = tokenized_dataset["validation"].shuffle(seed=123).select(range(200))
dataset_test = tokenized_dataset["test"].shuffle(seed=123).select(range(200))

dataloader_train = DataLoader(dataset_train, shuffle=True, batch_size=8)
dataloader_validation = DataLoader(dataset_validation, shuffle=True, batch_size=8)
dataloader_test = DataLoader(dataset_test, shuffle=True, batch_size=8)

# ------------------------------------------------
#            1.4 MODEL TRAINING
# ------------------------------------------------
optimizer = AdamW(model.parameters(), lr=2e-5)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"using device: {device}")
model.to(device)

def evaluate(dataloader):
  model.eval()

  labels = []
  predictions = []

  for batch in dataloader:
      batch = {k: v.to(device) for k, v in batch.items()}
      with torch.no_grad():
          outputs = model(**batch)

      logits = outputs.logits

      # copy tensors to cpu for metric calc
      labels.append(batch['labels'].cpu())
      predictions.append(torch.argmax(logits, dim=-1).cpu())

  labels = np.concatenate(labels)
  predictions = np.concatenate(predictions)

  f1 = f1_score(labels, predictions)
  accuracy = accuracy_score(labels, predictions)

  return f1, accuracy

def train():
  model.train()
  num_epochs = 4# TODO: set back to 4
  num_training_steps = num_epochs * len(dataloader_train)
  progress_bar = tqdm(range(num_training_steps))

  loss_history = list()
  f1_history = list()
  accuracy_history = list()

  best_f1 = 0

  for epoch in range(num_epochs):
    for batch in dataloader_train:
      batch = {k: v.to(device) for k, v in batch.items()}
      outputs = model(**batch)
      loss = outputs.loss

      loss_history.append(loss.item())

      loss.backward()

      optimizer.step()
      optimizer.zero_grad()
      progress_bar.update(1)

    # evaluate model after each epoch
    f1, accuracy = evaluate(dataloader_validation)
    f1_history.append(f1)
    accuracy_history.append(accuracy)

    if f1 > best_f1:
      print(f"epoch's f1 ({f1} is higher than best_f1 ({best_f1}) -> saving model")
      model.save_pretrained('model_checkpoint/')
      best_f1 = f1

  return loss_history, f1_history, accuracy_history


loss_history, f1_history, accuracy_history = train()

# loss per training step
fig = plt.figure()
plt.plot(loss_history)
#fig.suptitle('test title')
plt.xlabel('training step')
plt.ylabel('loss')
fig.savefig('training_loss_plot.pdf')

# f1 per epoch
fig = plt.figure()
plt.plot(f1_history)
#fig.suptitle('test title')
plt.xlabel('epoch')
plt.ylabel('val f1 score')
fig.savefig('validation_f1_score_plot.pdf')

# acc per epoch
fig = plt.figure()
plt.plot(accuracy_history)
#fig.suptitle('test title')
plt.xlabel('epoch')
plt.ylabel('val accuracy')
fig.savefig('validation_accuracy_plot.pdf')

# ------------------------------------------------
#            1.5 MODEL EVALUATION
# ------------------------------------------------
f1, accuracy = evaluate(dataloader_test)
print(f"final results on test: f1={f1}, acc={accuracy}")
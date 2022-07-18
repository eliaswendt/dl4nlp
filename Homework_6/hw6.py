from typing import OrderedDict
import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AdamW, AutoModelForQuestionAnswering, PretrainedConfig

from datasets import load_dataset
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
from matplotlib import pyplot as plt


def print_task_1_3_1(dataset):

  # iterate over the first three examples
  for idx in range(3):
    print(f"Example {idx}\n\tcontext: {dataset['train']['context'][idx]}")

    # every example can have multiple questions
    for qa in dataset['train'][idx]['qas']:
      print(f"\tquestion: {qa['question']}")

      # every question can have multiple answers
      for aidx, answer in enumerate(qa['answers']):
        print(f"\tanswer {aidx}: {answer}")





if __name__ == '__main__':
  # loading the dataset
  dataset = load_dataset('json', data_files={
    'train': 'datasets/train_TriviaQA-web.jsonl_converted.jsonl', 
    'test': 'datasets/dev_TriviaQA-web.jsonl_converted.jsonl'
  })
  print(dataset)

  torch.manual_seed(0) # set manual seed for reproducibility

  model_key = "huawei-noah/TinyBERT_General_4L_312D"
  model = AutoModelForQuestionAnswering.from_pretrained(model_key)
  tokenizer = AutoTokenizer.from_pretrained(model_key)
  #exit(0)

  #print_task_1_3_1(dataset)

  def tokenization_w_truncation(text):
    return tokenizer(text, padding="max_length", max_length=512, truncation=True)

  def tokenization_wo_truncation(text):
    return tokenizer(text, padding="max_length", max_length=512, truncation=False)

  dataset['train'] = dataset['train'].map(tokenization_w_truncation, batched=True)
  dataset['train']['question'] = dataset['train']['context'].map(tokenization_w_truncation, batched=True)
  dataset['train']['context'] = dataset['train']['context'].map(tokenization_w_truncation, batched=True)
  dataset['train']['context'] = dataset['train']['context'].map(tokenization_w_truncation, batched=True)
  dataset['train']['context'] = dataset['train']['context'].map(tokenization_w_truncation, batched=True) # use map function to quickly tokenize batches of examples
 
  print(tokenized_dataset)

  # original hyper parameters
  original_hps = {
    'train_batch_size': 16,
    'eval_batch_size': 32,
    'epochs': 1,
    'weight_decay': 0.01,
    'optimizer': AdamW(model.parameters(), lr=1e-5),
    'n_warm_up_steps': 0
  }

  device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
  print(f"using device: {device}")
  model.to(device)


  tokenized_dataset.set_format("torch")

  dataset_train = tokenized_dataset["train"].shuffle(seed=123).select(range(2000))
  dataset_validation = tokenized_dataset["validation"].shuffle(seed=123).select(range(200))

  dataloader_train = DataLoader(dataset_train, shuffle=True, batch_size=8)
  dataloader_validation = DataLoader(dataset_validation, shuffle=True, batch_size=8)

  def train(dataloader):
    model.train()
    num_epochs = 4# TODO: set back to 4
    num_training_steps = num_epochs * len(dataloader)
    progress_bar = tqdm(range(num_training_steps))

    loss_history = list()
    f1_history = list()
    accuracy_history = list()

    best_f1 = 0

    for epoch in range(num_epochs):
      for batch in dataloader:
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





from typing import OrderedDict
import json
import gzip
import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AdamW, AutoModelForQuestionAnswering, PretrainedConfig

from datasets import load_dataset
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
from matplotlib import pyplot as plt



def print_task_1_3_1(dataset):

  for idx in range(3):
    print(f'Example {idx}:')
    print('\tContext (capped at 100 chars): ', dataset[idx]["context"][:100])
    print('\tQuestion: ', dataset[idx]["question"])
    print('\tAnswer: ', dataset[idx]["answers"])

def evaluate(dataloader):
  model.eval()

  labels = []
  predictions = []

  for batch in dataloader:
      print(batch)
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

def train(dataloader, config):
  print('starting training')
  model.train()
  num_training_steps = config['epochs'] * len(dataloader)
  progress_bar = tqdm(range(num_training_steps))

  loss_history = list()
  f1_history = list()
  accuracy_history = list()

  best_f1 = 0

  for epoch in range(config['epochs']):
    for batch in dataloader:
      batch = {k: v.to(device) for k, v in batch.items()}
      outputs = model(**batch)
      loss = outputs.loss

      loss_history.append(loss.item())

      loss.backward()

      config['optimizer'].step()
      config['optimizer'].zero_grad()
      progress_bar.update(1)

    # # evaluate model after each epoch
    # f1, accuracy = evaluate(dataloader['test'])
    # f1_history.append(f1)
    # accuracy_history.append(accuracy)

    # if f1 > best_f1:
    #   print(f"epoch's f1 ({f1} is higher than best_f1 ({best_f1}) -> saving model")
    #   model.save_pretrained('model_checkpoint/')
    #   best_f1 = f1

  #return loss_history, f1_history, accuracy_history


def get_labels(inputs, answers):
  start_positions = []
  end_positions = []

  for i, offset in enumerate(inputs["offset_mapping"]):
      sample_idx = inputs["overflow_to_sample_mapping"][i]
      answer = answers[sample_idx]
      start_char = answer["answer_start"][0]
      end_char = answer["answer_start"][0] + len(answer["text"][0])
      sequence_ids = inputs.sequence_ids(i)

      # Find the start and end of the context
      idx = 0
      while sequence_ids[idx] != 1:
          idx += 1
      context_start = idx
      while sequence_ids[idx] == 1:
          idx += 1
      context_end = idx - 1

      # If the answer is not fully inside the context, label is (0, 0)
      if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
          start_positions.append(0)
          end_positions.append(0)
      else:
          # Otherwise it's the start and end token positions
          idx = context_start
          while idx <= context_end and offset[idx][0] <= start_char:
              idx += 1
          start_positions.append(idx - 1)

          idx = context_end
          while idx >= context_start and offset[idx][1] >= end_char:
              idx -= 1
          end_positions.append(idx + 1)

  return start_positions, end_positions


if __name__ == '__main__':
  torch.manual_seed(0) # set manual seed for reproducibility
  model_checkpoint = "huawei-noah/TinyBERT_General_4L_312D"

  # loading the dataset
  raw_datasets = load_dataset('json', data_files={
    #'train': 'datasets/train_TriviaQA-web.jsonl_converted.jsonl', 
    'test': 'datasets/dev_TriviaQA-web.jsonl_converted.jsonl'
  })

  #print_task_1_3_1(datasets['test']) # TODO: change back to 'train'
  
  #dataset = dataset.rename_column("answers", "labels")
  print(raw_datasets['test'])


  # print(f"len questions: {len(raw_datasets['test']['question'])}")
  # print(f"len input_ids: {len(raw_datasets['test']['input_ids'])}")

  # current issue: The error you have is due to the input_ids column not having the same number of examples as the other columns.
  tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
  print(f'tokenizer.is_fast: {tokenizer.is_fast}')
  def tokenization(dataset):
    return tokenizer(
      dataset['question'], 
      dataset['context'], 
      padding="max_length", 
      max_length=512, 
      truncation="only_second",
      return_overflowing_tokens=True,
      return_offsets_mapping=True
    )
  
  # map iterates over train and test
  inputs = raw_datasets.map(tokenization, batched=True)
  #print(tokenizer.decode(inputs['test'][0]))

  exit()

  # remove columns with strings
  inputs = inputs.remove_columns(['id', 'context', 'question'])
  print(inputs['test'][0].keys())

  start_positions, end_positions = get_labels(inputs['test'], raw_datasets['test']['answers'])
  print(start_positions, end_positions)

  exit()

  # load model checkpoint
  model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)

  #print(dataset)

  # original hyper parameters
  original_config = {
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

  inputs.set_format("torch")

  #dataset_train = tokenized_dataset["train"].shuffle(seed=123).select(range(2000))
  #dataset_validation = tokenized_dataset["validation"].shuffle(seed=123).select(range(200))


  #dataloader_train = DataLoader(dataset['train'])
  dataloader_test = DataLoader(inputs['test'], shuffle=True, batch_size=8)
  print(dataloader_test)
  train(dataloader_test, original_config)

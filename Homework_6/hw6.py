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
    'train': 'datasets/train_TriviaQA-web.jsonl.gz_prepared.jsonl', 
    'test': 'datasets/dev_TriviaQA-web.jsonl.gz_prepared.jsonl'
  })

  torch.manual_seed(0) # set manual seed for reproducibility

  # # TODO: register dataset in google sheet
  model_key = "huawei-noah/TinyBERT_General_4L_312D"
  model = AutoModelForQuestionAnswering.from_pretrained(model_key)
  tokenizer = AutoTokenizer.from_pretrained(model_key)

  print_task_1_3_1(dataset)

  def tokenization(file):
    return tokenizer(file['context'], file[''], padding="max_length", max_length=512, truncation=True) # TODO: truncation only on the context

  tokenized_dataset = dataset.map(tokenization, batched=True) # use map function to quickly tokenize batches of examples

  print(tokenized_dataset)
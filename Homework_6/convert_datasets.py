import json
import copy
import gzip
from tqdm.auto import tqdm

def convert_dataset(file_in, file_out):

  for line in tqdm(file_in):
    example = json.loads(line)

    # skip header
    if 'header' in example:
      continue

    context = example['context']
    for qa in example['qas']:
      id = qa['id']
      question = qa['question']
      
      answers = {
        'text': [],
        'answer_start': []
      }
      
      for answer in qa['detected_answers']:
        answers['text'].append(answer['text'])
        answers['answer_start'].append(answer['char_spans'][0][0])

      new_example = json.dumps({
        'id': id,
        'context': context,
        'question': question,
        'answers': answers
      })

      file_out.write(f'{new_example}\n')

if __name__ == '__main__':
  filepaths = [
    'datasets/train_TriviaQA-web.jsonl',
    'datasets/dev_TriviaQA-web.jsonl',
  ]

  #prepare_dataset(filepath)

  for filepath in filepaths:

    try:
      if filepath.split('.')[-1] == 'gz':
        file_in = gzip.open(filepath, 'rb')
      else:
        file_in = open(filepath)
      
      file_out = open(f'{filepath}_converted.jsonl', 'w')

    except:
      print(f'Could not open input "{filepath}" or output "{filepath}_converted.jsonl"')
      exit(1)

    convert_dataset(file_in, file_out)
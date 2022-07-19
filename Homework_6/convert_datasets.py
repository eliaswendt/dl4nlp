import json
import copy
import gzip
from tqdm.auto import tqdm

# DEPRECATED
def prepare_dataset(filepath):
  '''
  DEPRECATED
  Converts all integers in 
  ['qas'][:][''question_tokens'][1] AND
  ['context_tokens'][:][1[]
  to strings
  '''

  if filepath.split('.')[-1] == 'gz':
    file_in = gzip.open(filepath, 'rb')
  else:
    file_in = open(filepath)

  with open(f'{filepath}_prepared.jsonl', 'w') as file_out:

    for example in file_in:
      context = json.loads(example)
      if 'header' in context:
          continue

      original_qas = context['qas']
      new_qas = []

      for original_qas_entry in original_qas:
        new_qas_entry = copy.deepcopy(original_qas_entry)
        original_question_tokens = original_qas_entry['question_tokens']
        new_question_tokens = []
        for original_question_token in original_question_tokens:
          original_question_token[1] = str(original_question_token[1])

          new_question_tokens.append(original_question_token)

        new_qas_entry['question_tokens'] = new_question_tokens

      new_qas.append(new_qas_entry)
      context['qas'] = new_qas


      new_context_tokens = []
      for context_token in context['context_tokens']:
        context_token[1] = str(context_token[1])
        new_context_tokens.append(context_token)

      context['context_tokens'] = new_context_tokens

      file_out.write(f'{json.dumps(context)}\n')
      
  file_in.close()

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
    'datasets/dev_TriviaQA-web.jsonl',
    'datasets/train_TriviaQA-web.jsonl'
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
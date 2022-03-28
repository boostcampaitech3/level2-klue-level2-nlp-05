import pickle as pickle
import os
import pandas as pd
import torch


class RE_Dataset(torch.utils.data.Dataset):
  """ Dataset 구성을 위한 class."""
  def __init__(self, pair_dataset, labels):
    self.pair_dataset = pair_dataset
    self.labels = labels

  def __getitem__(self, idx):
    item = {key: val[idx].clone().detach() for key, val in self.pair_dataset.items()}
    item['labels'] = torch.tensor(self.labels[idx])
    return item

  def __len__(self):
    return len(self.labels)

def preprocessing_dataset(dataset):
  """ 처음 불러온 csv 파일을 원하는 형태의 DataFrame으로 변경 시켜줍니다."""
  subject_entity = []
  object_entity = []
  for i,j in zip(dataset['subject_entity'], dataset['object_entity']):
    subject_entity.append(eval(i)['word'])
    object_entity.append(eval(j)['word'])
  out_dataset = pd.DataFrame({'id':dataset['id'], 'sentence':dataset['sentence'],'subject_entity':subject_entity,'object_entity':object_entity,'label':dataset['label'],})
  return out_dataset

def eng2kor(type_name):
  if type_name == 'ORG':
    type_name = '조직'
  elif type_name == 'PER':
    type_name = '사람'
  elif type_name == 'LOC':
    type_name = '장소'
  elif type_name == 'DAT':
    type_name = '시간'
  elif type_name == 'POH':
    type_name = '고유명사'
  elif type_name == 'NOH':
    type_name = '숫자'

def preprocessing_dataset_typed_entity(dataset, eng=True):
  '''
  sentence에 punctuation으로 구분되는 typed entity marker를 추가합니다.
  eng=Ture는 영어, eng=False는 한글 entity marker를 추가합니다.
  '''
  # 사용하려면, load_daata 함수에서 dataset = preprocessing_dataset_typed_entity(pd_dataset)으로 바꿔주세요!
  subject_entity = []
  object_entity = []
  sentence_typed = []
  for subj, obj, sent in zip(dataset['subject_entity'], dataset['object_entity'], dataset['sentence']):
    subj = eval(subj)
    obj = eval(obj)

    subj_s = subj['start_idx']
    subj_e = subj['end_idx']
    obj_s = obj['start_idx']
    obj_e = obj['end_idx']
    subj_type = subj['type']
    obj_type = obj['type']

    if not eng:
      subj_type = eng2kor(subj_type)
      obj_type = eng2kor(obj_type)
            
    if subj_s < obj_s:
      sent_typed = sent[:subj_s]+'#^'+subj_type+'^'+sent[subj_s:subj_e+1]+'#'+sent[subj_e+1:obj_s]+'@*'+obj_type+'*'+sent[obj_s:obj_e+1]+'@'+sent[obj_e+1:]
    elif obj_s < subj_s:
      sent_typed = sent[:obj_s]+'#^'+obj_type+'^'+sent[obj_s:obj_e+1]+'#'+sent[obj_e+1:subj_s]+'@*'+subj_type+'*'+sent[subj_s:subj_e+1]+'@'+sent[subj_e+1:]
    else:
      sent_typed = sent

    subject_entity.append(subj['word'])
    object_entity.append(obj['word'])
    sentence_typed.append(sent_typed)
        
  out_dataset = pd.DataFrame({'id':dataset['id'], 'sentence':sentence_typed,'subject_entity':subject_entity,'object_entity':object_entity,'label':dataset['label'],})
  return out_dataset

def load_data(dataset_dir):
  """ csv 파일을 경로에 맡게 불러 옵니다. """
  pd_dataset = pd.read_csv(dataset_dir)
  # dataset = preprocessing_dataset(pd_dataset)
  dataset = preprocessing_dataset_typed_entity(pd_dataset)
  
  return dataset

def tokenized_dataset(dataset, tokenizer):
  """ tokenizer에 따라 sentence를 tokenizing 합니다."""
  concat_entity = []
  for e01, e02 in zip(dataset['subject_entity'], dataset['object_entity']):
    temp = ''
    temp = e01 + '[SEP]' + e02
    concat_entity.append(temp)
  tokenized_sentences = tokenizer(
      concat_entity,
      list(dataset['sentence']),
      return_tensors="pt",
      padding=True,
      truncation=True,
      max_length=256,
      add_special_tokens=True,
      )
  return tokenized_sentences

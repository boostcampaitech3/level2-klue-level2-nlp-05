import numpy as np
import torch
from torch.utils.data import Dataset, Subset, random_split
from transformers import AutoTokenizer
import pandas as pd
import pickle


class KLUEDataset(Dataset):
    def __init__(self, data_dir, model_name, task = None, val_ratio=0.1):
        self.data_dir = data_dir
        self.val_ratio = val_ratio
        self.task = task
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.tokenized_data, labels = self.setup()
        self.labels = self.label_to_num(labels)
        self.num_classes = 30

    def setup(self):
        subject_entity = []
        object_entity = []
        sentence_typed = []
        dataset = pd.read_csv(self.data_dir)

        for subj, obj, sent in zip(dataset['subject_entity'], dataset['object_entity'], dataset['sentence']):
            subj = eval(subj)
            obj = eval(obj)
            subject_entity.append(subj['word'])
            object_entity.append(obj['word'])
        
            if self.task == 'entity_marker':      
                subj_s = subj['start_idx']
                subj_e = subj['end_idx']
                obj_s = obj['start_idx']
                obj_e = obj['end_idx']
                subj_type = subj['type']
                obj_type = obj['type']
                subj_type = self.eng2kor(subj_type)
                obj_type = self.eng2kor(obj_type)
                
                if subj_s < obj_s:
                    sent_typed = sent[:subj_s]+' # '+sent[subj_s:subj_e+1]+' ^ '+subj_type+' ^ # '+sent[subj_e+1:obj_s]+' @ '+sent[obj_s:obj_e+1]+' * '+obj_type+' * @ '+sent[obj_e+1:]
                elif obj_s < subj_s:
                    sent_typed = sent[:obj_s]+' @ '+sent[obj_s:obj_e+1]+' * '+obj_type+' * @ '+sent[obj_e+1:subj_s]+' # '+sent[subj_s:subj_e+1]+' ^ '+subj_type+' ^ # '+sent[subj_e+1:]
                else:
                    sent_typed = sent
                sentence_typed.append(sent_typed)
        
        if self.task=='entity_marker':
            dataset = pd.DataFrame({'id':dataset['id'], 'sentence':sentence_typed,'subject_entity':subject_entity,'object_entity':object_entity,'label':dataset['label']})
            
            tokenized_sentences = self.tokenizer(
            list(dataset['sentence']),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
            add_special_tokens=True,
            return_token_type_ids=False
            )
            e1_masks = []
            e2_masks = []
            for sent in tokenized_sentences['input_ids']:
                sent = list(sent)
                e1_mask = [0]*512
                e2_mask = [0]*512
                e1_mask[sent.index(387)+1] = 1   # '#' 토큰 번호 찾아서 바꿔주기 현재는 mt5 모델 토큰id로 지정
                e2_mask[sent.index(1250)+1] = 1  # '@' 토큰 번호 찾아서 바꿔주기 현재는 mt5 모델 토큰id로 지정
                e1_masks.append(e1_mask)
                e2_masks.append(e2_mask)
            tokenized_sentences['e1_mask'] = torch.tensor(e1_masks)
            tokenized_sentences['e2_mask'] = torch.tensor(e2_masks)
        
        elif self.task =='multi_sentence':
            concat_entity = []
            for e01, e02 in zip(dataset['subject_entity'], dataset['object_entity']):
                temp = ''
                temp = '이 문장에서 ' + e01 + '와 ' + e02 + '는 어떤 관계일까?'
                concat_entity.append(temp)
            tokenized_sentences = self.tokenizer(
                list(dataset['sentence']),
                concat_entity,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=300,
                add_special_tokens=True,
                )

        else:
            dataset = pd.DataFrame({'id':dataset['id'], 'sentence':dataset['sentence'],'subject_entity':subject_entity,'object_entity':object_entity,'label':dataset['label']})
            concat_entity = []
            for e01, e02 in zip(dataset['subject_entity'], dataset['object_entity']):
                temp = ''
                temp = e01 + '[SEP]' + e02
                concat_entity.append(temp)
            tokenized_sentences = self.tokenizer(
                concat_entity,
                list(dataset['sentence']),
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256,
                add_special_tokens=True,
                )
        
        return tokenized_sentences, dataset['label']

    @staticmethod
    def eng2kor(type_name):
        typed_name_dict = {'ORG':'조직', 'PER':'사람', 'LOC':'장소', 'DAT':'시간', 'POH':'고유명사', 'NOH':'숫자'}
        return typed_name_dict[type_name]
    
    @staticmethod
    def label_to_num(label):
        num_label = []
        with open('/opt/ml/code/dict_label_to_num.pkl', 'rb') as f:
            dict_label_to_num = pickle.load(f)
        for v in label:
            num_label.append(dict_label_to_num[v])

        return num_label

    def __getitem__(self, index): 
        item = {key: val[index].clone().detach() for key, val in self.tokenized_data.items()}
        labels = torch.tensor(self.labels[index])
        return item, labels

    def __len__(self):
        return len(self.labels)

    def split_dataset(self):
        """
        데이터셋을 train 과 val 로 나눕니다,
        pytorch 내부의 torch.utils.data.random_split 함수를 사용하여
        torch.utils.data.Subset 클래스 둘로 나눕니다.
        구현이 어렵지 않으니 구글링 혹은 IDE (e.g. pycharm) 의 navigation 기능을 통해 코드를 한 번 읽어보는 것을 추천드립니다^^
        """
        n_val = int(len(self) * self.val_ratio)
        n_train = len(self) - n_val
        train_set, val_set = random_split(self, [n_train, n_val])
        return train_set, val_set
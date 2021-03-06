from tkinter import TRUE
import numpy as np
import torch
from torch.utils.data import Dataset, Subset, random_split
from transformers import AutoTokenizer
import pandas as pd
import pickle


class KLUEDataset(Dataset):
    def __init__(self, data_dir, model_name, task=None, val_ratio=0.1, mode='train'):
        self.data_dir = data_dir
        self.val_ratio = val_ratio
        self.task = task
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.mode = mode
        self.tokenized_data, self.dataset = self.setup()
        if self.mode == 'train':
            self.labels = self.label_to_num(self.dataset['label'].values)
        self.ids = self.dataset['id'].values
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
                    sent_typed = sent[:subj_s] + ' # ' + sent[subj_s:subj_e + 1] + ' ^ ' + subj_type + ' ^ # ' + \
                                 sent[subj_e + 1:obj_s] + ' @ ' + sent[obj_s:obj_e + 1] + ' * ' + obj_type + ' * @ ' + \
                                 sent[obj_e + 1:]
                elif obj_s < subj_s:
                    sent_typed = sent[:obj_s] + ' @ ' + sent[obj_s:obj_e + 1] + ' * ' + obj_type + ' * @ ' + \
                                 sent[obj_e + 1:subj_s] + ' # ' + sent[subj_s:subj_e + 1] + ' ^ ' + subj_type + ' ^ # ' \
                                 + sent[subj_e + 1:]
                else:
                    sent_typed = sent
                sentence_typed.append(sent_typed)

        if self.task == 'entity_marker':
            dataset = pd.DataFrame({'id': dataset['id'], 'sentence': sentence_typed, 'subject_entity': subject_entity,
                                    'object_entity': object_entity, 'label': dataset['label']})

            tokenized_sentences = self.tokenizer(
                list(dataset['sentence']),
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
                add_special_tokens=True,
                return_token_type_ids=True
            )
            e1_masks = []
            e2_masks = []
            if 't5' in self.model_name:
                tok1 = 387
                tok2 = 1250
            elif 'bigbird' in self.model_name:
                tok1 = 507
                tok2 = 536
            elif 'bert' in self.model_name or 'electra' in self.model_name:
                tok1 = 7
                tok2 = 36

            for sent in tokenized_sentences['input_ids']:
                sent = list(sent)
                e1_mask = [0] * 512
                e2_mask = [0] * 512
                e1_mask[sent.index(tok1) + 1] = 1  # '#' ?????? ?????? ????????? ???????????? ????????? mt5 ?????? ??????id??? ??????
                e2_mask[sent.index(tok2) + 1] = 1  # '@' ?????? ?????? ????????? ???????????? ????????? mt5 ?????? ??????id??? ??????
                e1_masks.append(e1_mask)
                e2_masks.append(e2_mask)
            tokenized_sentences['e1_mask'] = torch.tensor(e1_masks)
            tokenized_sentences['e2_mask'] = torch.tensor(e2_masks)

        elif self.task == 'multi_sentence':
            concat_entity = []
            for e01, e02 in zip(dataset['subject_entity'], dataset['object_entity']):
                temp = ''
                temp = '??? ???????????? ' + e01 + '??? ' + e02 + '??? ?????? ?????????????'
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
            dataset = pd.DataFrame(
                {'id': dataset['id'], 'sentence': dataset['sentence'], 'subject_entity': subject_entity,
                 'object_entity': object_entity, 'label': dataset['label']})
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

        return tokenized_sentences, dataset

    @staticmethod
    def eng2kor(type_name):
        typed_name_dict = {'ORG': '??????', 'PER': '??????', 'LOC': '??????', 'DAT': '??????', 'POH': '????????????', 'NOH': '??????'}
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
        if self.mode == 'train':
            labels = torch.tensor(self.labels[index])
            return item, labels
        else: # if self.mode == 'eval':
            return item

    def __len__(self):
        return len(self.dataset)

    def split_dataset(self):
        """
        ??????????????? train ??? val ??? ????????????,
        pytorch ????????? torch.utils.data.random_split ????????? ????????????
        torch.utils.data.Subset ????????? ?????? ????????????.
        ????????? ????????? ????????? ????????? ?????? IDE (e.g. pycharm) ??? navigation ????????? ?????? ????????? ??? ??? ???????????? ?????? ??????????????????^^
        """
        n_val = int(len(self) * self.val_ratio)
        n_train = len(self) - n_val
        train_set, val_set = random_split(self, [n_train, n_val])
        return train_set, val_set
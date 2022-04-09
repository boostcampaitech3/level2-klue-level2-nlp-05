import os
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import DataLoader
from importlib import import_module
import pandas as pd
import torch
import torch.nn.functional as F

import pickle as pickle
import numpy as np
import argparse
from tqdm import tqdm

from dataset import KLUEDataset
from model import *


def load_model(device, model_folder: str, MODEL_NAME: str, num_classes=30):
    model_config = AutoConfig.from_pretrained(MODEL_NAME)
    model_config.num_labels = num_classes
    model_module = getattr(import_module("model"), args.model_class)
    model = model_module(
        config=model_config,
        model_name=MODEL_NAME
    ).to(device)
    # model = MT5ForTypedEntityMarker(config=model_config,
    #                                 model_name = MODEL_NAME)
    model_path = os.path.join(model_folder, 'best', 'best.pth')  # TODO: 가장 최근 실험으로 하던.. 모델명/폴더명 위치를 바꿔야할듯?
    print(model_path)
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    return model


def inference(model, tokenized_sent, device):
    """
    test dataset을 DataLoader로 만들어 준 후,
    batch_size로 나눠 model이 예측 합니다.
    """
    dataloader = DataLoader(tokenized_sent, batch_size=16, shuffle=False)
    model.eval()
    output_pred = []
    output_prob = []
    for i, data in enumerate(tqdm(dataloader)):
        inputs = data

        with torch.no_grad():
            args_dict = {}
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)
            args_dict['input_ids'] = input_ids
            args_dict['attention_mask'] = attention_mask

            if args.model_name != 't5':
                token_type_ids = inputs['token_type_ids'].to(device)
                args_dict['token_type_ids'] = token_type_ids

            if args.task == 'entity_marker':
                e1_mask = inputs['e1_mask'].to(device)
                e2_mask = inputs['e2_mask'].to(device)
                args_dict['e1_mask'] = e1_mask
                args_dict['e2_mask'] = e2_mask

            outputs = model(**args_dict)
        logits = outputs
        prob = F.softmax(logits, dim=-1).detach().cpu().numpy()
        logits = logits.detach().cpu().numpy()
        result = np.argmax(logits, axis=-1)

        output_pred.append(result)
        output_prob.append(prob)

    return np.concatenate(output_pred).tolist(), np.concatenate(output_prob, axis=0).tolist()

def num_to_label(label):
    """
    숫자로 되어 있던 class를 원본 문자열 라벨로 변환 합니다.
    """
    origin_label = []
    with open('dict_num_to_label.pkl', 'rb') as f:
        dict_num_to_label = pickle.load(f)
    for v in label:
        origin_label.append(dict_num_to_label[v])

    return origin_label


def main(args, model_folder, release_folder):
    """
    주어진 dataset csv 파일과 같은 형태일 경우 inference 가능한 코드입니다.
    """
    # -- settings
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(device)

    # -- model name setting
    if args.model_name == 't5':
        MODEL_NAME = 'google/mt5-base'
    elif args.model_name == 'bert':
        MODEL_NAME = 'klue/bert-base'
    elif args.model_name == 'kobigbird':
        MODEL_NAME = 'monologg/kobigbird-bert-base'
    elif args.model_name == 'koelectra':
        MODEL_NAME = 'monologg/koelectra-small-v3-discriminator'
    else:
        MODEL_NAME = 'klue/roberta-large'

    # -- dataset
    test_dataset = KLUEDataset(data_dir=os.path.join(args.dataset_folder, args.test_dataset),  # 데이터셋 경로 지정
                               model_name=MODEL_NAME,
                               task=args.task,
                               mode=args.mode)
    test_id = test_dataset.ids
    # test_label = list(map(int,test_dataset.labels))
    print('Dataset load success')

    # -- model
    model = load_model(device=device, model_folder=model_folder, MODEL_NAME=MODEL_NAME)

    ## predict answer
    pred_answer, output_prob = inference(model, test_dataset, device)  # model에서 class 추론
    pred_answer = num_to_label(pred_answer)  # 숫자로 된 class를 원래 문자열 라벨로 변환.

    ## make csv file with predicted answer
    #########################################################
    # 아래 directory와 columns의 형태는 지켜주시기 바랍니다.
    output = pd.DataFrame({'id': test_id, 'pred_label': pred_answer, 'probs': output_prob, })

    output.to_csv(os.path.join(release_folder, 'submission.csv'), index=False)  # 최종적으로 완성된 예측한 라벨 csv 파일 형태로 저장.
    #### 필수!! ##############################################
    print('---- Finish! ----')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # -- Dataset Folder
    parser.add_argument('--dataset_folder', type=str, default='/opt/ml/dataset/test', help='dataset root folder')
    parser.add_argument('--test_dataset', type=str, default='test_data.csv', help='test dataset path')

    # -- model setting
    # parser.add_argument('--model_name', type=str, default='klue/bert-base', help='huggingface model name')
    parser.add_argument('--mode', type=str, default='eval', help='train, eval')
    parser.add_argument('--task', type=str, default='entity_marker', help='entity_marker, multi_sentence, None')
    parser.add_argument('--model_name', type=str, default='t5', help='huggingface model name')
    parser.add_argument('--model_class', type=str, default='MT5ForTypedEntityMarker')

    # -- save setting
    parser.add_argument('--model_dir', type=str, default="/workspace/lv2-klue/src/models")
    parser.add_argument('--output_dir', type=str, default='/workspace/lv2-klue/src/outputs')

    args = parser.parse_args()

    # -- set model folder
    model_folder = os.path.join(args.model_dir, args.model_name)

    # -- set release folder
    release_folder = os.path.join(args.output_dir, args.model_name)
    try:
        if not os.path.exists(release_folder):
            os.makedirs(release_folder)
            print(f'create {release_folder}')
    except OSError:
        print(f'ERROR occurred creating directory {release_folder}')

    main(args, model_folder, release_folder)
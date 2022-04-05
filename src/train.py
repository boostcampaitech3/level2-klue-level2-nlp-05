import argparse
import os
import glob
import re
import random
import json
import gc
from importlib import import_module
from pathlib import Path

import pickle as pickle
import numpy as np
import pandas as pd

import sklearn
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, precision_recall_curve, auc

import torch
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from transformers import AutoTokenizer, AutoConfig
from datetime import datetime

import wandb

from dataset import KLUEDataset


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def increment_path(path, exist_ok=False):
    """ Automatically increment path, i.e. runs/exp --> runs/exp0, runs/exp1 etc.
    Args:
        path (str or pathlib.Path): f"{model_dir}/{args.name}".
        exist_ok (bool): whether increment path (increment if False).
    """
    path = Path(path)
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}*")
        matches = [re.search(rf"%s(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]
        n = max(i) + 1 if i else 2
        return f"{path}{n}"


def klue_re_micro_f1(preds, labels):
    """KLUE-RE micro f1 (except no_relation)"""
    label_list = ['no_relation', 'org:top_members/employees', 'org:members',
                  'org:product', 'per:title', 'org:alternate_names',
                  'per:employee_of', 'org:place_of_headquarters', 'per:product',
                  'org:number_of_employees/members', 'per:children',
                  'per:place_of_residence', 'per:alternate_names',
                  'per:other_family', 'per:colleagues', 'per:origin', 'per:siblings',
                  'per:spouse', 'org:founded', 'org:political/religious_affiliation',
                  'org:member_of', 'per:parents', 'org:dissolved',
                  'per:schools_attended', 'per:date_of_death', 'per:date_of_birth',
                  'per:place_of_birth', 'per:place_of_death', 'org:founded_by',
                  'per:religion']
    no_relation_label_idx = label_list.index("no_relation")
    label_indices = list(range(len(label_list)))
    label_indices.remove(no_relation_label_idx)
    return f1_score(labels, preds, average="micro", labels=label_indices) * 100.0


def klue_re_auprc(probs, labels):
    """KLUE-RE AUPRC (with no_relation)"""
    labels = np.eye(30)[labels]

    score = np.zeros((30,))
    for c in range(30):
        targets_c = labels.take([c], axis=1).ravel()
        preds_c = probs.take([c], axis=1).ravel()
        precision, recall, _ = precision_recall_curve(targets_c, preds_c)
        score[c] = auc(recall, precision)
    return np.average(score) * 100.0


def compute_metrics(outs, preds, labels):
    """ validation을 위한 metrics function """

    # calculate accuracy using sklearn's function
    outs = outs.detach().cpu().numpy()
    preds = preds.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()

    f1 = klue_re_micro_f1(preds, labels)
    auprc = klue_re_auprc(outs, labels)
    acc = accuracy_score(labels, preds)  # 리더보드 평가에는 포함되지 않습니다.

    return {
        'micro f1 score': f1,
        'auprc': auprc,
        'accuracy': acc,
    }


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def label_to_num(label):
    num_label = []
    with open('/opt/ml/level2-klue/baseline/dict_label_to_num.pkl', 'rb') as f:
        dict_label_to_num = pickle.load(f)
    for v in label:
        num_label.append(dict_label_to_num[v])

    return num_label


def train(model_folder, args):
    seed_everything(args.seed)
    np.seterr(invalid='ignore')

    # --- wandb setting
    wandb.init(project='level2-klue', entity='team-oeanhdoejo')
    USER_NAME = args.user_name
    dt_string = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    wandb.config.update(args)
    wandb.run.name = f"{USER_NAME}-{args.model_name}-{args.epochs}-{args.batch_size}-{dt_string}"  # set run format
    wandb.run.save()

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
    dataset = KLUEDataset(data_dir=os.path.join(args.dataset_folder, args.train_dataset),
                          model_name=MODEL_NAME,
                          task=args.task,
                          val_ratio=0.1)
    num_classes = dataset.num_classes

    if not args.valid_dataset:
        train_dataset, valid_dataset = dataset.split_dataset()
    else:
        train_dataset = dataset
        valid_dataset = KLUEDataset(data_dir=os.path.join(args.dataset_folder, args.valid_dataset),
                            model_name=MODEL_NAME,
                            task=args.task)
    print('Dataset load success')

    # -- data loader
    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              pin_memory=use_cuda,
                              drop_last=True,
                              )
    valid_loader = DataLoader(valid_dataset,
                              batch_size=args.batch_size,
                              pin_memory=use_cuda,
                              drop_last=True,
                              )

    # -- model
    model_config = AutoConfig.from_pretrained(MODEL_NAME)
    model_config.num_labels = num_classes

    model_module = getattr(import_module("model"), args.model_class)
    model = model_module(
        config=model_config,
        model_name=MODEL_NAME,
    ).to(device)

    # -- loss & metric
    criterion = torch.nn.CrossEntropyLoss()  # CrossEntropy fix
    optimizer = torch.optim.AdamW(params=model.parameters(),
                                  lr=args.learning_rate,
                                  weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, args.lr_decay_step, gamma=0.5)

    # -- logging
    save_dir = increment_path(os.path.join(model_folder, args.name))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f'create {save_dir}')
    logger = SummaryWriter(log_dir=save_dir)

    with open(os.path.join(save_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=4)

    # -- training start
    best_val_f1 = 0
    best_val_loss = np.inf

    for epoch in range(args.epochs):
        # train loop
        model.train()
        loss_value = 0
        matches = 0
        for idx, train_batch in enumerate(train_loader):
            inputs, labels = train_batch

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
            labels = labels.to(device)

            optimizer.zero_grad()
            outs = model(**args_dict)
            
            preds = torch.argmax(outs, dim=-1)
            loss = criterion(outs, labels)
            matches += (preds == labels).sum().item()

            loss.backward()
            optimizer.step()

            loss_value += loss.item()

            if (idx + 1) % args.log_interval == 0:
                train_loss = loss_value / args.log_interval
                # train_acc = matches / args.batch_size / args.log_interval
                metrics_dict = compute_metrics(outs, preds, labels)
                current_lr = get_lr(optimizer)
                print(
                    f"Epoch[{epoch}/{args.epochs}]({idx + 1}/{len(train_loader)}) || "
                    f"training loss {train_loss:4.4} || training micro f1 score {metrics_dict['micro f1 score']:0.2f} || lr {current_lr}"
                    # f"training loss {train_loss:4.4} || training accuracy {train_acc:4.2%} || lr {current_lr}"
                )
                logger.add_scalar("Train/loss", train_loss, epoch * len(train_loader) + idx)
                # logger.add_scalar("Train/accuracy", train_acc, epoch * len(train_loader) + idx)

                loss_value = 0
                matches = 0

                # wandb.log({"train": {"acc": train_acc, "loss": train_loss}}, step=epoch)  # wandb add
                wandb.log({"train": {"micro f1 score": metrics_dict['micro f1 score'], "loss": train_loss}}, step=epoch)  # wandb add

        scheduler.step()

        with torch.no_grad():
            print("Calculating validation results...")
            model.eval()
            val_loss_items = []
            for valid_batch in valid_loader:
                inputs, labels = valid_batch

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
                labels = labels.to(device)

                outs = model(**args_dict)
                preds = torch.argmax(outs, dim=-1)

                loss_item = criterion(outs, labels).item()
                val_loss_items.append(loss_item)

            val_loss = np.sum(val_loss_items) / len(valid_loader)
            best_val_loss = min(best_val_loss, val_loss)
            metrics_dict = compute_metrics(outs, preds, labels)
            val_f1 = metrics_dict['micro f1 score']

            if val_f1 > best_val_f1:
                print(f"New best model for val accuracy : {val_f1:0.2f}! saving the best model..")
                best_val_f1 = max(best_val_f1, metrics_dict['micro f1 score'])
                torch.save(model.state_dict(), f"{save_dir}/best.pth")
            torch.save(model.state_dict(), f"{save_dir}/last.pth")
            print(
                f"[Val] f1 : {val_f1:0.2f}, loss: {val_loss:4.2} || "
                f"best f1 : {best_val_f1:0.2f}, best loss: {best_val_loss:4.2}"
            )
            logger.add_scalar("Val/loss", val_loss, epoch)
            logger.add_scalar("Val/f1", val_f1, epoch)
            print()

        wandb.log({"valid": {"f1": val_f1, "loss": val_loss}}) # wandb.add
    wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # -- Dataset Folder
    parser.add_argument('--dataset_folder', type=str, default='/opt/ml/dataset/train', help='dataset root folder')
    parser.add_argument('--name', default='exp', help='model save at {MODEL_DIR}/{name}')
    parser.add_argument('--valid_dataset', type=str, default='split_dev_processed.csv', help='validation dataset path')
    parser.add_argument('--train_dataset', type=str, default='train_BT_preprocessed.csv',
                        help='train dataset path')


    # -- hyper parameters
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--metric', type=str, default='micro f1 score')
    parser.add_argument('--strategy', type=str, default='epoch')
    parser.add_argument('--learning_rate', type=int, default=5e-5)
    parser.add_argument('--warmup_steps', type=int, default=500)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--log_interval', type=int, default=20, help='how many batches to wait before logging training status')
    parser.add_argument('--lr_decay_step', type=int, default=20,
                        help='learning rate scheduler deacy step (default: 20)')
    # -- wandb setting
    parser.add_argument('--user_name', type=str, default='Minji', help='wandb user name')

    # -- model setting
    # parser.add_argument('--model_name', type=str, default='klue/bert-base', help='huggingface model name')
    parser.add_argument('--model_name', type=str, default='t5', help='huggingface model name')
    parser.add_argument('--task', type=str, default='entity_marker', help='entity_marker, multi_sentence, None')
    parser.add_argument('--model_class', type=str, default='MT5ForTypedEntityMarker')

    # -- save setting
    parser.add_argument('--model_dir', type=str, default="./models")
    parser.add_argument('--output_dir', type=str, default='./results')

    args = parser.parse_args()

    # -- create model folder
    model_folder = os.path.join(args.model_dir, args.model_name)
    try:
        if not os.path.exists(model_folder):
            os.makedirs(model_folder)
            print(f'create {model_folder}')
    except OSError:
        print(f'ERROR occurred creating directory {model_folder}')

    # -- empty cache
    gc.collect()
    torch.cuda.empty_cache() # empty cache before train loop

    train(model_folder, args)

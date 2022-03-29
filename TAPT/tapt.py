from transformers import AutoConfig, BertTokenizer, BertForPreTraining, BertForMaskedLM, RobertaForMaskedLM, AdamW
import torch
from tqdm import tqdm
import wandb
from datetime import datetime


class PretrainDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
    def __getitem__(self, idx):
        return {key:torch.tensor(val[idx]) for key, val in self.encodings.items()}
    def __len__(self):
        return len(self.encodings.input_ids)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

MODEL_NAME = 'klue/bert-base'
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

# setting model hyperparameter
model_config =  AutoConfig.from_pretrained(MODEL_NAME)
model_config.num_labels = 30

model =  BertForMaskedLM.from_pretrained(MODEL_NAME, config=model_config)
model.to(device)

with open('./pretrain_corpus.txt', 'r') as fp:
    text = fp.read().split('\n')
inputs = tokenizer(text, return_tensors='pt', max_length=512, truncation=True, 
                   padding='max_length')
inputs['labels'] = inputs.input_ids.detach().clone()

rand = torch.rand(inputs.input_ids.shape)
mask_arr = (rand < 0.15) * (inputs.input_ids != 2) * (inputs.input_ids != 3) * (inputs.input_ids != 0)

selection = []

for i in range(mask_arr.shape[0]):
    selection.append(
        torch.flatten(mask_arr[i].nonzero()).tolist()
    )
for i in range(mask_arr.shape[0]):
    inputs.input_ids[i, selection[i]] = 4

dataset = PretrainDataset(inputs)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

wandb.init(project="level2-klue", entity="team-oeanhdoejo")
NAME = 'TAPT'
dt_string = datetime.now().strftime("%d%m_%H")
wandb.run.name = f"{NAME}-{MODEL_NAME}-{dt_string}"

torch.cuda.empty_cache()
model.train()

optim = AdamW(model.parameters(), lr=1e-5)

epochs = 5

for epoch in range(epochs):
    loop = tqdm(dataloader, leave=True)
    for batch in loop:
        optim.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask,
                        labels=labels)
        loss = outputs.loss
        loss.backward()
        optim.step()

        loop.set_description(f'Epoch {epoch}')
        loop.set_postfix(loss=loss.item())

model.save_pretrained('./pretrained_model')
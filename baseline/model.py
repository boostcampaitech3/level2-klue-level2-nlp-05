import torch
import torch.nn as nn
from transformers import RobertaModel, RobertaPreTrainedModel


class FCLayer(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate=0.0, use_activation=True):
        super(FCLayer, self).__init__()
        self.use_activation = use_activation
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.dropout(x)
        if self.use_activation:
            return self.relu(self.linear(x))
        else:
            return self.linear(x)


class TypedEntityRoberta(RobertaPreTrainedModel):
    def __init__(self, config):
        super(TypedEntityRoberta, self).__init__(config)
        self.roberta = RobertaModel(config=config)  # Load pretrained roberta

        self.num_labels = config.num_labels

        self.entity_fc_layer = FCLayer(config.hidden_size*2, config.hidden_size)
        self.label_classifier = FCLayer(
            config.hidden_size,
            config.num_labels,
            use_activation=False,
        )

    @staticmethod
    def entity_hidden(hidden_output, e_mask):
        """
        Get entity hidden state for the first token of entity subtokens
        :param hidden_output: [batch_size, sequence_length, dim]
        :param e_mask: [batch_size, max_seq_len]
                e.g. e_mask[0] == [0, 0, 0, 1, 1, 1, 0, 0, ... 0]
        :return: [batch_size, dim]
        """
        e_mask_unsqueeze = e_mask.unsqueeze(1) 
        e_mask_unsqueeze = e_mask_unsqueeze[:,:,:hidden_output.size(dim=1)]
        length_tensor = (e_mask != 0).sum(dim=1).unsqueeze(1)  # [batch_size, 1]

        # [b, 1, sequence_length] * [b, sequence_length, dim] = [b, 1, dim] -> [b, dim]
        vector = torch.bmm(e_mask_unsqueeze.float(), hidden_output).squeeze(1)
        return vector

    def forward(self, input_ids, attention_mask, token_type_ids, labels, e1_mask, e2_mask):
        outputs = self.roberta(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )  # sequence_output, pooled_output, (hidden_states), (attentions)
        sequence_output = outputs[0]
        pooled_output = outputs[1]  # [CLS]

        e1_h = self.entity_hidden(sequence_output, e1_mask)
        e2_h = self.entity_hidden(sequence_output, e2_mask)
        concat_h = torch.cat([e1_h, e2_h], dim=-1)
        
        z = self.entity_fc_layer(concat_h)
        logits = self.label_classifier(z)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        # Softmax
        if labels is not None:
            if self.num_labels == 1:
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)

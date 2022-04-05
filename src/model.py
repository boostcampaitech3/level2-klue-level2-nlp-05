import torch
import torch.nn as nn
from transformers import RobertaModel, BertModel, MT5EncoderModel, BigBirdModel, ElectraModel


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


class BertForSequenceClassification(nn.Module):
    def __init__(self, config, model_name):
        super().__init__()
        self.encoder = BertModel.from_pretrained(model_name, config=config)
        self.label_classifier = FCLayer(
            config.hidden_size,
            config.num_labels,
            use_activation=False,
        )

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.encoder(input_ids, attention_mask, token_type_ids)
        logits = self.label_classifier(outputs[1])
        return logits


class RobertaForSequenceClassification(nn.Module):
    def __init__(self, config, model_name):
        super().__init__()
        self.encoder = RobertaModel.from_pretrained(model_name, config=config)
        self.label_classifier = FCLayer(
            config.hidden_size,
            config.num_labels,
            use_activation=False,
        )

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.encoder(input_ids, attention_mask, token_type_ids)
        logits = self.label_classifier(outputs[1])
        return logits


class MT5ForSequenceClassification(nn.Module):
    def __init__(self, config, model_name):
        super().__init__()
        self.encoder = MT5EncoderModel.from_pretrained(model_name, config=config)
        self.label_classifier = FCLayer(
            config.hidden_size,
            config.num_labels,
            use_activation=False,
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids, attention_mask)
        logits = self.label_classifier(outputs.last_hidden_state[:, 0, :])
        return logits


class BertForTypedEntityMarker(nn.Module):
    def __init__(self, config, model_name):
        super().__init__()
        self.encoder = BertModel.from_pretrained(model_name, config=config)  # Load pretrained roberta

        self.num_labels = config.num_labels

        self.entity_fc_layer = FCLayer(config.hidden_size * 2, config.hidden_size)
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
        e_mask_unsqueeze = e_mask_unsqueeze[:, :, :hidden_output.size(dim=1)]
        # [b, 1, sequence_length] * [b, sequence_length, dim] = [b, 1, dim] -> [b, dim]
        vector = torch.bmm(e_mask_unsqueeze.float(), hidden_output).squeeze(1)
        return vector

    def forward(self, input_ids, attention_mask, token_type_ids, labels, e1_mask, e2_mask):
        outputs = self.encoder(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )  # sequence_output, pooled_output, (hidden_states), (attentions)
        sequence_output = outputs[0]

        e1_h = self.entity_hidden(sequence_output, e1_mask)
        e2_h = self.entity_hidden(sequence_output, e2_mask)
        concat_h = torch.cat([e1_h, e2_h], dim=-1)

        z = self.entity_fc_layer(concat_h)
        logits = self.label_classifier(z)

        return logits


class RobertaForTypedEntityMarker(nn.Module):
    def __init__(self, config, model_name):
        super().__init__()
        self.encoder = RobertaModel.from_pretrained(model_name, config=config)  # Load pretrained roberta

        self.num_labels = config.num_labels

        self.entity_fc_layer = FCLayer(config.hidden_size * 2, config.hidden_size)
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
        e_mask_unsqueeze = e_mask_unsqueeze[:, :, :hidden_output.size(dim=1)]
        # [b, 1, sequence_length] * [b, sequence_length, dim] = [b, 1, dim] -> [b, dim]
        vector = torch.bmm(e_mask_unsqueeze.float(), hidden_output).squeeze(1)
        return vector

    def forward(self, input_ids, attention_mask, token_type_ids, labels, e1_mask, e2_mask):
        outputs = self.encoder(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )  # sequence_output, pooled_output, (hidden_states), (attentions)
        sequence_output = outputs[0]

        e1_h = self.entity_hidden(sequence_output, e1_mask)
        e2_h = self.entity_hidden(sequence_output, e2_mask)
        concat_h = torch.cat([e1_h, e2_h], dim=-1)

        z = self.entity_fc_layer(concat_h)
        logits = self.label_classifier(z)

        return logits


class MT5ForTypedEntityMarker(nn.Module):
    def __init__(self, config, model_name):
        super().__init__()
        self.encoder = MT5EncoderModel.from_pretrained(model_name, config=config)

        self.cls_fc_layer = FCLayer(config.hidden_size, config.hidden_size, 0.0)
        self.entity_fc_layer = FCLayer(config.hidden_size * 2, config.hidden_size, 0.0)
        self.label_classifier = FCLayer(
            config.hidden_size,
            config.num_labels,
            0.0,
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
        e_mask_unsqueeze = e_mask_unsqueeze[:, :, :hidden_output.size(dim=1)]
        # [b, 1, sequence_length] * [b, sequence_length, dim] = [b, 1, dim] -> [b, dim]
        vector = torch.bmm(e_mask_unsqueeze.float(), hidden_output).squeeze(1)
        return vector

    def forward(self, input_ids, attention_mask, e1_mask, e2_mask):
        outputs = self.encoder(
            input_ids, attention_mask=attention_mask
        )  # sequence_output, pooled_output, (hidden_states), (attentions)

        sequence_output = outputs[0]
        # Average
        e1_h = self.entity_hidden(sequence_output, e1_mask)
        e2_h = self.entity_hidden(sequence_output, e2_mask)
        concat_h = torch.cat([e1_h, e2_h], dim=-1)

        z = self.entity_fc_layer(concat_h)
        logits = self.label_classifier(z)
        return logits


class ElectraForTypedEntityMarker(nn.Module):
    def __init__(self, config, model_name):
        super().__init__()
        self.encoder = ElectraModel.from_pretrained(model_name, config=config)

        self.cls_fc_layer = FCLayer(config.hidden_size, config.hidden_size, 0.0)
        self.entity_fc_layer = FCLayer(config.hidden_size * 2, config.hidden_size, 0.0)
        self.label_classifier = FCLayer(
            config.hidden_size,
            config.num_labels,
            0.0,
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
        e_mask_unsqueeze = e_mask_unsqueeze[:, :, :hidden_output.size(dim=1)]
        # [b, 1, sequence_length] * [b, sequence_length, dim] = [b, 1, dim] -> [b, dim]
        vector = torch.bmm(e_mask_unsqueeze.float(), hidden_output).squeeze(1)
        return vector

    def forward(self, input_ids, attention_mask, token_type_ids, e1_mask, e2_mask):
        outputs = self.encoder(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )  # sequence_output, pooled_output, (hidden_states), (attentions)

        sequence_output = outputs[0]
        # Average
        e1_h = self.entity_hidden(sequence_output, e1_mask)
        e2_h = self.entity_hidden(sequence_output, e2_mask)
        concat_h = torch.cat([e1_h, e2_h], dim=-1)

        z = self.entity_fc_layer(concat_h)
        logits = self.label_classifier(z)
        return logits


class BigBirdForTypedEntityMarker(nn.Module):
    def __init__(self, config, model_name):
        super().__init__()
        self.encoder = BigBirdModel.from_pretrained(model_name, config=config)

        self.cls_fc_layer = FCLayer(config.hidden_size, config.hidden_size, 0.0)
        self.entity_fc_layer = FCLayer(config.hidden_size * 2, config.hidden_size, 0.0)
        self.label_classifier = FCLayer(
            config.hidden_size,
            config.num_labels,
            0.0,
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
        e_mask_unsqueeze = e_mask_unsqueeze[:, :, :hidden_output.size(dim=1)]
        # [b, 1, sequence_length] * [b, sequence_length, dim] = [b, 1, dim] -> [b, dim]
        vector = torch.bmm(e_mask_unsqueeze.float(), hidden_output).squeeze(1)
        return vector

    def forward(self, input_ids, attention_mask, token_type_ids, e1_mask, e2_mask):
        outputs = self.encoder(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )  # sequence_output, pooled_output, (hidden_states), (attentions)

        sequence_output = outputs[0]
        # Average
        e1_h = self.entity_hidden(sequence_output, e1_mask)
        e2_h = self.entity_hidden(sequence_output, e2_mask)
        concat_h = torch.cat([e1_h, e2_h], dim=-1)

        z = self.entity_fc_layer(concat_h)
        logits = self.label_classifier(z)
        return logits
import torch

import torch.nn as nn
import torch.hub as pretrained

from typing import Dict, Tuple
from functools import partial
from transformers.models.bert import BertModel

from code.config import Hyper
from code.models.classifier import Classifier
from code.metrics import F1, Indicator


class AEModel(nn.Module):
    def __init__(self, hyper: Hyper):
        super(AEModel, self).__init__()
        self.gpu = hyper.gpu

        self.encoder = pretrained.load('huggingface/pytorch-transformers', 'model', 'bert-base-cased')
        # self.encoder = BertModel.from_pretrained('bert-base-cased')

        embed_dim = self.encoder.config.hidden_size
        self.entity_embedding = nn.Embedding(hyper.entity_vocab_size, embed_dim)
        self.event_embedding = nn.Embedding(hyper.event_vocab_size, embed_dim)

        self.pool = partial(torch.mean, dim=-2)
        self.pooling = lambda x, valid_cnt: self.pool(x) * x.shape[-2] / valid_cnt
        
        self.main_classifier = Classifier(embed_dim, hyper.role_vocab_size)

        self.main_loss = nn.CrossEntropyLoss()

        self.main_metric = F1(hyper)
        self.get_metric = self.main_metric.report
        self.role_indicator = Indicator(hyper)
        # self.NonRole_indicator = Indicator(hyper, {0: [1, 0], 1: [0, 1]})

        self.to(hyper.gpu)

    def reset(self) -> None:
        self.main_metric.reset()

    def forward(self, sample, is_train: bool=False) -> Dict:
        output = {}
        labels = sample.label.cuda(self.gpu)

        entity_encoding, trigger_encoding = self._forward_encoder(sample, is_train)
        logits = self.main_classifier(entity_encoding, trigger_encoding)

        output['loss'] = self.main_loss(logits, target=labels)
        
        if is_train:
            output["description"] = partial(self.description, output=output)
        else:
            self._update_metric(logits, labels)
            output["probability"] = torch.softmax(logits, dim=-1)
            
        return output

    def _forward_encoder(self, sample, is_train: bool) -> Tuple:
        entity_id, event_id = sample.entity_id.cuda(self.gpu), sample.event_id.cuda(self.gpu)
        entity_embedding = self.entity_embedding(entity_id)
        event_embedding = self.event_embedding(event_id)

        text_id = sample.tokens.cuda(self.gpu)
        bert_mask = torch.gt(text_id, 0).long()
        bert_mask.requires_grad = False
        segment = torch.zeros_like(bert_mask)
        bert_embedding = self.encoder.embeddings(input_ids=text_id, token_type_ids=segment)
        
        embedding = bert_embedding + entity_embedding + event_embedding

        bert_output = None
        if is_train:
            self.encoder.train()
            bert_output = self.encoder(inputs_embeds=embedding, attention_mask=bert_mask, output_hidden_states=False)
        else:
            with torch.no_grad():
                bert_output = self.encoder(inputs_embeds=embedding, attention_mask=bert_mask, output_hidden_states=False)
        h = bert_output.last_hidden_state[:, 1:-1, :]

        entity_encoding = self._pooling_multi_tokens(h, sample.entity_start.cuda(self.gpu), sample.entity_end.cuda(self.gpu))
        trigger_encoding = self._pooling_multi_tokens(h, sample.trigger_start.cuda(self.gpu), sample.trigger_end.cuda(self.gpu))

        return entity_encoding, trigger_encoding
        
    def _pooling_multi_tokens(self, tensor, start, end) -> torch.tensor:
        masked_tensor, valid_cnt = self._slice_tensor_from_start_to_end(tensor, start, end)
        return self.pooling(masked_tensor, valid_cnt)

    def _slice_tensor_from_start_to_end(self, tensor, start, end) -> Tuple:
        batch_size, seq_len, _ = tensor.shape
        serial_matrix = torch.arange(0, seq_len).repeat(batch_size, 1).cuda(self.gpu)
        start, end = start.view(-1, 1), end.view(-1, 1)
        start_mask = torch.ge(serial_matrix, start)
        end_mask = torch.lt(serial_matrix, end)
        mask = start_mask * end_mask
        mask = mask.float()
        mask = mask.unsqueeze(-1)
        mask.requires_grad = False
        valid_cnt = end - start
        valid_cnt = valid_cnt.float()
        valid_cnt.requires_grad = False
        masked_tensor = tensor * mask
        return masked_tensor, valid_cnt

    @staticmethod
    def description(epoch, epoch_num, output) -> str:
        return "L: {:.2f}, epoch: {}/{}:".format(
            output["loss"].item(), epoch, epoch_num,
        )    

    def _update_metric(self, logits, labels) -> None:
        predicts = torch.argmax(logits, dim=-1)
        self.main_metric.update(golden_labels=labels.cpu(), predict_labels=predicts.cpu())
    
    def reset_indicators(self) -> None:
        self.role_indicator.reset()
        # self.NonRole_indicator.reset()

    def update_indicators(self, sample, prob):
        # to_binary = lambda x: torch.gt(x, 0).long()
        self.role_indicator.update(prob, sample.label, sample.entity_type)
        # self.NonRole_indicator.update(prob, to_binary(sample.label), to_binary(sample.entity_type))
    
    def report(self):
        return (
            self.main_metric.report_all(), 
            self.role_indicator.report(),
            # self.NonRole_indicator.report()
            None
        )
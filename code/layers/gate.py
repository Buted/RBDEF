import torch

import torch.nn as nn

from code.config import Hyper


class Gate(nn.Module):
    def __init__(self, embed_dim: int):
        super(Gate, self).__init__()
        self.gate_linear = nn.Linear(embed_dim*2, embed_dim)
        # self._init_linear()
    
    def _init_linear(self) -> None:
        nn.init.xavier_normal_(self.gate_linear.weight)
        self.gate_linear.bias.data.fill_(0)

    def forward(self, entity, trigger):
        h = torch.cat((entity, trigger), dim=-1)
        gate = torch.sigmoid(self.gate_linear(h))
        return gate * entity + (1 - gate) * trigger


class ScalableGate(nn.Module):
    def __init__(self, embed_dim: int, out_dim: int):
        super(ScalableGate, self).__init__()
        self.entity_scaling = nn.Linear(embed_dim, out_dim)
        self.trigger_scaling = nn.Linear(embed_dim, out_dim)
        self.gate = Gate(out_dim)

    def forward(self, entity, trigger):
        entity = self.entity_scaling(entity)
        trigger = self.trigger_scaling(trigger)
        return self.gate(entity, trigger)


# class ScalableGateWithEmbedding(nn.Module):
#     def __init__(self, embed_dim: int, hyper: Hyper):
#         super(ScalableGateWithEmbedding, self).__init__()
#         self.entity_embedding = nn.Embedding(hyper.entity_vocab_size, hyper.out_dim)
#         self.event_embedding = nn.Embedding(hyper.event_vocab_size, hyper.out_dim)
#         self.entity_scaling = nn.Linear(embed_dim, hyper.out_dim)
#         self.trigger_scaling = nn.Linear(embed_dim, hyper.out_dim)
#         self.gate = Gate(hyper.out_dim)

#     def forward(self, entity, trigger, entity_id, event_id):
#         entity_emb = self.entity_embedding(entity_id)
#         event_emb = self.event_embedding(event_id)
#         entity = self.entity_scaling(entity)
#         trigger = self.trigger_scaling(trigger)
#         entity = entity + entity_emb
#         trigger = trigger + event_emb
#         return self.gate(entity, trigger)


class ScalableGateWithEmbedding(nn.Module):
    def __init__(self, embed_dim: int, hyper: Hyper):
        super(ScalableGateWithEmbedding, self).__init__()
        self.entity_embedding = nn.Embedding(hyper.entity_vocab_size, hyper.out_dim)
        self.event_embedding = nn.Embedding(hyper.event_vocab_size, hyper.out_dim)
        self.entity_scaling = nn.Linear(embed_dim, hyper.out_dim)
        self.trigger_scaling = nn.Linear(embed_dim, hyper.out_dim)
        self.gate = Gate(hyper.out_dim)

    def forward(self, entity, trigger, entity_id, event_id):
        entity_emb = self.entity_embedding(entity_id)
        event_emb = self.event_embedding(event_id)
        entity = self.entity_scaling(entity)
        trigger = self.trigger_scaling(trigger)
        # entity = entity + entity_emb + event_emb
        # trigger = trigger + entity_emb + event_emb
        return self.gate(entity, trigger) + entity_emb + event_emb
        # return self.gate(entity, trigger)


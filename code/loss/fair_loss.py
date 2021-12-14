import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def fair_reg(preds, Xp, device):
    Xp = Xp.unsqueeze(1)
    Xp = Xp.type(torch.FloatTensor).to(device)
    Xp = torch.cat((Xp, 1-Xp), dim=1)
    viol = preds.mean()-(preds@Xp)/torch.max(Xp.sum(axis=0), torch.ones(Xp.shape[1]).to(device)*1e-5)
    return (viol**2).mean()


class LDAMLoss(nn.Module):
    def __init__(self, cls_num_list, max_m=0.5, s=30, device=None):
        '''
        code form https://github.com/kaidic/LDAM-DRW/blob/master/losses.py
        cls_num_list: the number of instances in each class, a list of numbers where cls_num_list[0] is the number of instances in class 0
            note: for ldam loss the last layer of the model should NOT be nn.Linear, it should be nn.NormedLinear
        '''
        super(LDAMLoss, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.FloatTensor(m_list).to(device)
        m_list.requires_grad = False
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.gpu = device

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.uint8).to(self.gpu)
        index.scatter_(1, target.data.view(-1, 1), 1)
        index_float = index.type(torch.FloatTensor).to(self.gpu)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0,1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m
    
        output = torch.where(index, x_m, x)
        # return F.cross_entropy(output, target)
        return F.cross_entropy(self.s*output, target)


class FairLDAMLoss(nn.Module):
    def __init__(self, device, cls_num_list, group_size, max_m=0.5, ldams=30, rho=0.0):
        '''
        cls_num_list: the number of instances in each class, a list of numbers where cls_num_list[0] is the number of instances in class 0
        clsp_num_list: the number of instances in each group
            note: for ldam loss the last layer of the model should NOT be nn.Linear, it should be nn.NormedLinear
        '''
        super(FairLDAMLoss, self).__init__()
        self.gpu = device
        self.ldam = LDAMLoss(device=device, cls_num_list=cls_num_list, max_m=max_m, s=ldams)

        self.group_size = group_size
        self.rho = rho

    def forward(self, x, target, group):
        ldam_loss = self.ldam(x, target)    
        
        regloss = self._fair_reg(x, group)

        # return ldam_loss
        return ldam_loss + self.rho * regloss

    def _fair_reg(self, x, group_label):
        loss = None
        x_mean = torch.mean(x, dim=0)
        for group in range(self.group_size):
            group_x = x[group_label==group]
            if group_x.shape[0]:
                group_x_mean = torch.mean(group_x, dim=0)
                group_loss = torch.sum((group_x_mean - x_mean) ** 2)
                if loss is None:
                    loss = group_loss
                else:
                    loss += group_loss
        return loss
    
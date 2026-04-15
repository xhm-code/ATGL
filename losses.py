import torch
import torch.nn as nn
import torch.nn.functional as F


class ATGLoss(nn.Module):
    def __init__(self, p_weight = -1):
        self.p_weight = p_weight
        super().__init__()

    def forward(self, logits, labels):
        labels = labels.clone()
        th_label = torch.zeros_like(labels, dtype=torch.float, device=labels.device)
        p_mask = labels.clone().bool()
        p_mask[:, 0] = 0
        n_mask = (~labels.bool())
        n_mask[:, 0] = 0

        th_label[p_mask] = self.p_weight
        th_label[n_mask] = 0.0
        th_label[:, 0] = 1.0

        loss = -(F.log_softmax(logits, dim=-1) * th_label).sum(1).mean()

        return loss

    def get_label(self, logits, num_labels=-1):
        th_logit = logits[:, 0].unsqueeze(1)
        output = torch.zeros_like(logits).to(logits)
        mask = (logits > th_logit)
        if num_labels > 0:
            top_v, _ = torch.topk(logits, num_labels, dim=1)
            top_v = top_v[:, -1]
            mask = (logits >= top_v.unsqueeze(1)) & mask
        output[mask] = 1.0
        output[:, 0] = (output.sum(1) == 0.).to(logits)
        return output

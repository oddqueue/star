from typing import Mapping, Text, Tuple

import torch


class ARLoss(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.target_vocab_size = config.model.vq_model.codebook_size
        self.criterion = torch.nn.CrossEntropyLoss(reduction="mean")

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, Mapping[Text, torch.Tensor]]:
        shift_logits = logits[..., :-1, :].permute(0, 2, 1).contiguous()  # NLC->NCL
        shift_labels = labels.contiguous()
        shift_logits = shift_logits.view(
            shift_logits.shape[0], self.target_vocab_size, -1
        )
        shift_labels = shift_labels.view(shift_labels.shape[0], -1)
        shift_labels = shift_labels.to(shift_logits.device)
        loss = self.criterion(shift_logits, shift_labels)
        correct_tokens = (
            (torch.argmax(shift_logits, dim=1) == shift_labels).sum(dim=1)
            / shift_labels.size(1)
        )
        return loss, {"loss": loss, "correct_tokens": correct_tokens.mean()}

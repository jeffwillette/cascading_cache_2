import torch


def pad_targets(inputs, targets, ignore_index=-100):
    """
    for causal LLM when the offset inputs and targets differ in length
    """
    if targets.size(1) < inputs.size(1):
        target_pad = torch.full(
            (inputs.size(0), inputs.size(1) - targets.size(1)),
            ignore_index,
            device=inputs.device,
            dtype=inputs.dtype,
        )

        targets = torch.cat((targets, target_pad), dim=-1)
    return targets

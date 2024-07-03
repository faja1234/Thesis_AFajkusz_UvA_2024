import torch

def semantic_loss(y_pred, y_true, rule, class_weight, sem_weight):
    """
    Custom loss function combining binary cross-entropy loss and rule-based penalties in the form of semantic loss.

    Parameters:
        y_pred (torch.Tensor): Predicted probabilities.
        y_true (torch.Tensor): True labels.
        rule (torch.Tensor): Provided logical rule.
        sem_weight (float): Weight of semantic loss.

    Returns:
        torch.Tensor: Loss value.
    """
    # Calculate the original loss
    criterion = torch.nn.BCELoss(weight=class_weight)
    loss = criterion(y_pred, y_true)

    # Calculate the rule-based penalty
    rule_penalty = -torch.log(torch.sum(rule*y_pred))

    # Combine original loss and semantic loss
    total_loss = loss + sem_weight * rule_penalty

    return total_loss

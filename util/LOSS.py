import torch
import torch.nn as nn
import torch.nn.functional as F 

class CrossModalContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1):
        """
        Cross-modal N-pair contrastive loss for aligning text and speech embeddings.

        Args:
            temperature: Scaling factor for softmax (default: 0.1)
        """
        super(CrossModalContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, text_embeddings, speech_embeddings):
        """
        Compute the contrastive loss.

        Args:
            text_embeddings: Tensor of shape (batch_size, seq_len_text, embed_dim)
            speech_embeddings: Tensor of shape (batch_size, seq_len_speech, embed_dim)

        Returns:
            Loss value (scalar).
        """

        # Normalize embeddings for cosine similarity
        text_pooled = F.normalize(text_embeddings, p=2, dim=1)  # (batch_size, embed_dim)
        speech_pooled = F.normalize(speech_embeddings, p=2, dim=1)  # (batch_size, embed_dim)

        # Compute cosine similarities (batch_size x batch_size)
        logits = torch.matmul(speech_pooled, text_pooled.T)  # Cosine similarity matrix
        logits /= self.temperature

        # Ground truth labels (positive pairs are along the diagonal)
        batch_size = logits.size(0)
        labels = torch.arange(batch_size).to(logits.device)

        # Compute cross-entropy loss
        loss = F.cross_entropy(logits, labels)

        return loss
       
class LabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing=0.001, num_classes=2):
        """
        Precise Label Smoothing Loss
        
        Args:
            smoothing (float): Smoothing parameter α
            num_classes (int): Number of classes K
        """
        super().__init__()
        self.smoothing = smoothing
        self.num_classes = num_classes
        self.confidence = 1.0 - smoothing
        self.smoothing_value = smoothing / (num_classes - 1)

    def forward(self, inputs, targets):
        """
        Apply precise label smoothing
        
        Args:
            inputs (torch.Tensor): Model predictions (logits or probabilities)
            targets (torch.Tensor): Binary ground truth labels
        
        Returns:
            torch.Tensor: Label smoothed loss
        """
        # For binary classification, we'll adapt the multi-class formula
        # Reshape targets to ensure correct dimensionality
        targets = targets.view(-1, 1)
        
        # Create smoothed targets according to the formula:
        # yLSu_k = y_k * (1 - α) + α / K
        smoothed_targets = targets * self.confidence + self.smoothing_value
        
        # Compute cross-entropy loss with smoothed targets
        loss_fct = nn.BCELoss()
        loss = loss_fct(inputs.view(-1), smoothed_targets.view(-1))
        
        return loss

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        """
        Focal Loss implementation
        
        Args:
            alpha (torch.Tensor, optional): Weight for each class. Useful for class imbalance
            gamma (float): Focusing parameter. Higher gamma means more focus on hard examples
            reduction (str): 'mean', 'sum' or 'none'
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)  # Probabilities for the correct class
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
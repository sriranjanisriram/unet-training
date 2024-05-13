# Author: Sriranjani Sriram

import torch

def iou_score(pred_mask, true_mask):
    """Calculate the Intersection over Union (IoU) score."""

    intersection = torch.logical_and(true_mask, pred_mask).sum().float()
    union = torch.logical_or(true_mask, pred_mask).sum().float()
    iou = intersection / union if union > 0 else 0.0
    return iou


class EarlyStopper:
    """Early stopping to prevent overfitting."""

    def __init__(self, patience=1, folder="models/"):
        """Initializes the early stopper."""

        self.patience = patience
        self.folder = folder
        self.counter = 0
        self.min_validation_loss = float("inf")

    def early_stop(self, validation_loss, model):
        """Checks if early stopping criteria is met.

        Args:
            validation_loss: Current validation loss.
            model: The model being trained.

        Returns:
            True if early stopping criteria is met, otherwise False.
        """

        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
            torch.save(model, self.folder + "best-model.pt")
            torch.save(
                model.state_dict(), self.folder + "best-model-parameters.pt"
            )
        elif validation_loss > self.min_validation_loss:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
from torch import nn

def loss_criterion(loss_type="Bce", weight=None):
    #TODO: Add metric lerning loss
    if loss_type=="Bce":
        """
        Numerical more stable than using Sigmoid + BCE Loss
        """
        criterion = nn.BCEWithLogitsLoss(weight=weight)

    elif loss_type=="Crossentropy":
        criterion = nn.CrossEntropyLoss(weight=weight)

    return criterion



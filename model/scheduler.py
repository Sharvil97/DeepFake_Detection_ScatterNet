from torch.optim import lr_scheduler 

def scheduler(optimizer, epoch, lr=None, scheduler_type="LambdaLR"):
    #TODO: Add other schedulers
    if scheduler_type == "LambdaLR":
        lambda1 = lambda epoch: epoch // 30
        lambda2 = lambda epoch: 0.95 ** epoch
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=[lambda1, lambda2])
    elif scheduler_type == "MultiplicativeLR":
        lmbda = lambda epoch: 0.95
        scheduler = lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lmbda)
    elif scheduler_type == "StepLR":
        scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    elif scheduler_type == "MultiStepLR":
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[30,80], gamma=0.1)
    elif scheduler_type == "ExponentialLR":
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.95, last_epoch=-1)
    # elif scheduler_type == "CosineAnnealingLR":
        # scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0, last_epoch=-1)
    elif scheduler_type == "ReduceLROnPlateau":
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    elif scheduler_type == "CyclicLR":
        scheduler = lr_scheduler.CyclicLR(optimizer, base_lr=lr, max_lr=0.1)
    return scheduler
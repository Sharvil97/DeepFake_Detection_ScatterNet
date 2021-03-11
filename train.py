import os
import gc
import time
import torch
import copy
import argparse
import datetime
from data_loader.data_loader import  ImageFolderTrain, ImageFolderVal
import torch_optimizer as toptim
from torch import optim
from torch.utils.data import DataLoader
from model.loss import loss_criterion
from model.scheduler import scheduler
from tensorboardX import SummaryWriter
from metric.metric import prec_rec
from model.model import ResScatterNet, ResidualBlock
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve, accuracy_score, f1_score


def train(model_name, dataset, num_epochs, loss_type, log_interval, num_classes, batch_size, optimizer, lr, scheduler_type,\
 augmentation, save_path, logdir, flush_history=False, load_model=None, parallel = True, val=True, return_best=True):
    
    training_time = time.time()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    train_dataset, train_loader, val_dataset, val_loader = load_dataloaders(dataset, batch_size, augmentation)

    now = datetime.now()
    logdir = logdir + now.strftime("%Y%m%d-%H%M%S") + "/"
    os.makedirs(logdir)
    log_root_folder = logdir #"../saved/log/"
    if flush_history == 1:
        objects = os.listdir(log_root_folder)
        for f in objects:
            if os.path.isdir(log_root_folder + f):
                shutil.rmtree(log_root_folder + f)
    #TensorboardX
    writer = SummaryWriter(logdir)

    #metrics
    average_auc = []
    average_loss = []
    average_acc = []
    average_ap = []
    average_one_rec = []
    average_five_rec = []
    average_nine_rec = []
    average_f1 = []
    best_acc = 0.0
    best_loss = 100
    current_acc = 0.0
    current_loss = 100.0
    best_auc = 0.0
    best_ap = 0.0
    best_f1 = 0.0

    if load_model is None:
        if model_name=="ResScatterNet":
            model = ResScatterNet(ResidualBlock, [2, 2, 2])
            if parallel:
                model = torch.nn.DataParallel(model).to(device)
            else:
                model = model.to(device)
    else:
        model = torch.load(load_model_path)
        if parallel:
            model = torch.nn.DataParallel(model).to(device)
        else:
            model = model.to(device)

        
    # if loss_type=="Bce":
    # # elif loss_type=="Crossentropy":
    criterion = loss_criterion(loss_type=loss_type)

    if optimizer=="Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer=="SGD":
        optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    elif optimizer=="RAdam":
        optimizer = toptim.RAdam(model.parameters(), lr= lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
    
    scheduler = scheduler(optimizer, epoch=num_epochs, lr=lr, scheduler_type=scheduler_type)

    if return_best:
        best_model_state = copy.deepcopy(model.state_dict())

    for e in range(num_epochs):
        print(f"Epoch: {e}/{num_epochs}")
        
        #running metrics
        running_loss = 0.0
        running_corrects = 0.0
        running_auc_labels = []
        running_ap_labels = []
        running_f1_labels = []
        running_auc_preds = []
        running_ap_preds = []
        runnung_f1_preds = []

        #training 
        model.train()

        for i, (imgs, labels) in enumerate(tqdm(train_loader)):
            imgs = imgs.to(device)
            labels = labels.to(device)
            # set accumulated gradients to zero
            optimizer.zero_grad()

            #forward pass
            with torch.set_grad_enabled(True):
                predictions = model(imgs)
                #predictions for accuracy
                thresh_preds = torch.round(torch.sigmoid(predictions))
                loss = loss_criterion(predictions.squeeze(-1), labels.type_as(predictions))
                loss.backwards()
                optimizer.step()

            running_loss += loss.item()*img.size(0)
            #calculating the accuracy
            running_corrects += torch.sum(thresh_preds==labels.unsqueeze(1))
            running_auc_labels.extend(labels.detach().cpu().numpy())
            running_auc_preds.extend(torch.sigmoid(predictions).detach().cpu().numpy())
            running_ap_labels.extend(labels.detach().cpu().numpy())
            running_ap_preds.extend(torch.sigmoid(predictions).detach().cpu().numpy())
            running_f1_labels.extend(labels.detach().cpu().numpy())
            runnung_f1_preds.extend(labels.detach().cpu().numpy())

            # for logging
            y_pred = torch.sigmoid(predictions).detach().cpu().numpy()
            y_true = labels.detach().cpu().numpy()

            #logging metrics for each step
            writer.add_scalar('Train/Acc', accuracy_score(y_true, y_pred), e*len(train_loader) + i)
            writer.add_scalar('Train/Loss', loss.item(), e*len(train_loader) + i)
            writer.add_scalar('Train/average_precision', average_precision_score(y_true, y_scores), e*len(train_loader) + i)
            writer.add_scalar('Train/AUC', roc_auc_score(y_true, y_scores), e*len(train_loader) + i)
            writer.add_scalar('Train/F1', f1_score(y_true, y_pred, 'weighted'), e*len(train_loader) + i)
            
            if (j%log_interval==0) & (j>0):
                print(f"Training: Loss: {loss.item():.4f} | Acc: {accuracy_score(y_true, y_pred):.4f} | AUC: {roc_auc_score(y_true, y_scores):.4f} | AP: {average_precision_score(y_true, y_scores):.4f} | F1: {f1_score(y_true, y_pred, 'weighted').:4f}")

        #step scheduler after each epoch
        scheduler.step()

        # epoch end metrics
        total_len = len(train_dataset)

        epoch_loss = running_loss/total_len
        epoch_acc = running_corrects/total_len
        epoch_auc = roc_auc_score(running_auc_labels, running_auc_preds)
        epoch_ap = average_precision_score(running_ap_labels, running_ap_preds)
        epoch_f1 = f1_score(running_f1_labels, runnung_f1_preds, 'weighted')

        print(f"Training: Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f} | AUC: {epoch_auc:.4f} | AP: {epoch_ap:.4f} | F1: {epoch_f1.:4f}")

        # logging to TensorboardX
        writer.add_scalar('Train/Epoch_Acc', epoch_acc, e + i)
        writer.add_scalar('Train/Epoch_Loss', epoch_loss, e + i)
        writer.add_scalar('Train/Epoch_average_precision', epoch_ap, e + i)
        writer.add_scalar('Train/Epoch_AUC', epoch_auc, e + i)
        writer.add_scalar('Train/Epoch_F1', epoch_f1, e + i)

        if e+1 == num_epochs:
            print("Saving model.")
            torch.save(model.state_dict(), f"{save_path}" + f"/{model_name}_{dataset}_epochs_{num_epochs}_optimizer_{optimizer}.pth")

        #Validation 
        model.eval()
        
        #running metrics
        val_running_loss = 0.0
        val_running_corrects = 0.0
        val_running_auc_labels = []
        val_running_ap_labels = []
        val_running_f1_labels = []
        val_running_auc_preds = []
        val_running_ap_preds = []
        val_runnung_f1_preds = []

        for j, (imgs, labels) in enumerate(tqdm(val_loader)):
            imgs = imgs.to(device)
            labels = labels.to(device)
            # set accumulated gradients to zero
            optimizer.zero_grad()

            #forward pass
            with torch.set_grad_enabled(False):
                predictions = model(imgs)
                #predictions for accuracy
                thresh_preds = torch.round(torch.sigmoid(predictions))
                loss = loss_criterion(predictions.squeeze(-1), labels.type_as(predictions))

            val_running_loss += loss.item()*img.size(0)
            #calculating the accuracy
            val_running_corrects += torch.sum(thresh_preds==labels.unsqueeze(1))
            val_running_auc_labels.extend(labels.detach().cpu().numpy())
            val_running_auc_preds.extend(torch.sigmoid(predictions).detach().cpu().numpy())
            val_running_ap_labels.extend(labels.detach().cpu().numpy())
            val_running_ap_preds.extend(torch.sigmoid(predictions).detach().cpu().numpy())
            val_running_f1_labels.extend(labels.detach().cpu().numpy())
            val_runnung_f1_preds.extend(labels.detach().cpu().numpy())

            # for logging
            y_pred = torch.sigmoid(predictions).detach().cpu().numpy()
            y_true = labels.detach().cpu().numpy()

            #logging metrics for each step
            writer.add_scalar('Val/Acc', accuracy_score(y_true, y_pred), e*len(train_loader) + j)
            writer.add_scalar('Val/Loss', loss.item(), e*len(train_loader) + j)
            writer.add_scalar('Val/average_precision', average_precision_score(y_true, y_scores), e*len(train_loader) + j)
            writer.add_scalar('Val/AUC', roc_auc_score(y_true, y_scores), e*len(train_loader) + j)
            writer.add_scalar('Val/F1', f1_score(y_true, y_pred, 'weighted'), e*len(train_loader) + j)

            if (j%log_interval==0) & (j>0):
                print(f"Validation: Loss: {loss.item():.4f} | Acc: {accuracy_score(y_true, y_pred):.4f} | AUC: {roc_auc_score(y_true, y_scores):.4f} | AP: {average_precision_score(y_true, y_scores):.4f} | F1: {f1_score(y_true, y_pred, 'weighted').:4f}")
                # torch.save(model.state_dict(), f"{save_path}" + f"/{model_name}_{dataset}_epochs_{num_epochs}_optimizer_{optimizer}_step_{}.pth")



        # epoch end metrics
        total_len = len(val_dataset)

        epoch_loss = running_loss/total_len
        epoch_acc = running_corrects/total_len
        epoch_auc = roc_auc_score(running_auc_labels, running_auc_preds)
        epoch_ap = average_precision_score(running_ap_labels, running_ap_preds)
        epoch_f1 = f1_score(running_f1_labels, runnung_f1_preds, 'weighted')

        print(f"Validation: Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f} | AUC: {epoch_auc:.4f} | AP: {epoch_ap:.4f} | F1: {epoch_f1.:4f}")

        # logging to TensorboardX
        writer.add_scalar('Val/Epoch_Acc', epoch_acc, e + j)
        writer.add_scalar('Val/Epoch_Loss', epoch_loss, e + j)
        writer.add_scalar('Val/Epoch_average_precision', epoch_ap, e + j)
        writer.add_scalar('Val/Epoch_AUC', epoch_auc, e + j)
        writer.add_scalar('Val/Epoch_F1', epoch_f1, e + j)


        #save the best model every epoch
        if epoch_acc > best_acc or (epoch_acc==best_acc and epoch_loss<best_loss):
            print("Saving a better model.") 
            one_rec, five_rec, nine_rec = metrics.prec_rec(val_running_auc_labels, val_running_auc_preds, method, alpha=100, plot=False)
            best_acc = epoch_acc
            best_loss = epoch_loss
            best_auc = epoch_auc
            best_ap = epoch_ap
            best_f1 = epoch_f1

            #save
            torch.save(model.state_dict(), f"{save_path}" + f"/{model_name}_{dataset}_epochs_{num_epochs}_optimizer_{optimizer}_acc_{epoch_acc}.pth")

            print(f"Best Validation: Loss: {best_loss:.4f} | Acc: {best_acc:.4f} | AUC: {best_auc:.4f} | AP: {best_ap:.4f} | F1: {best_f1.:4f}")

            if return_best:
                best_model_state = copy.deepcopy(model.state_dict())

        else:
            print("Model not saved.")
            print(f"Best Validation: Loss: {best_loss:.4f} | Acc: {best_acc:.4f} | AUC: {best_auc:.4f} | AP: {best_ap:.4f} | F1: {best_f1.:4f}")

        gc.collect()
        torch.cuda.empty_cache()
        
        average_acc.append(best_acc)
        average_ap.append(best_ap)
        average_auc.append(best_auc)
        average_loss.append(best_loss)
        average_f1.append(best_f1)
        average_one_rec.append(one_rec)
        average_five_rec.append(five_rec)
        average_nine_rec.append(nine_rec)

    writer.close()

    #print average results
    print("Best Average Metrics")
    average_auc = np.array(average_auc).mean()
    average_ap = np.array(average_ap).mean()
    average_acc = np.mean(np.asarray([acc.cpu().numpy() for acc in average_acc]))
    average_loss = np.mean(np.asarray([loss for loss in average_loss]))
    average_f1 = np.array(average_f1).mean()
    average_one_rec = np.array(average_one_rec).mean()
    average_five_rec = np.array(average_five_rec).mean()
    average_nine_rec = np.array(average_nine_rec).mean()

    #print average metrics
    print(f"Average best metrics: Loss: {average_loss:.4f} | Acc: {average_acc:.4f} | AUC: {average_auc:.4f} | AP: {average_ap:.4f} |\
         F1: {average_f1.:4f}")
    print(f"{average_one_rec} cost for 0.1 recall.")
    print(f"{average_five_rec} cost for 0.5 recall.")
    print(f"{average_nine_rec} cost for 0.9 recall.")
    print(f"Time Required: {(time.time() - training_time) // 60} min and {(time.time() - training_time) % 60} sec.")

    if return_best:
        model.load_state_dict(best_model_state)

    return model, average_auc, average_ap, average_acc, average_loss, average_f1

def load_dataloaders(dataset, batch_size, augmentations):
    train_dataset = ImageFolderTrain(dataset, augmentations=augmentations)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = ImageFolderVal(dataset, augmentations=augmentations)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    return train_dataset, train_loader, val_dataset, val_loader


def run(args):
    train(args.model_name, args.dataset, args.num_epochs, args.loss_type, args.log_interval, args.num_classes, args.batch_size, args.optimizer,\
        args.lr, args.scheduler_type, args.augmentation, args.save_path, args.logdir, args.flush_history, args.load_model,\
        args.parallel, args.val, args.return_best)


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_name', type=str, default='Deepfake',  required=True,\
        help="Provide the name of model used to train.")
    parser.add_argument('--dataset', type=str, default='FaceForensics', choices=['FaceForensics', 'FaceForensics++', 'CelebDF', 'GoogleDFD'\
        ,'FaceHQ', 'DFDC', 'DeeperForensics', 'UADFV'], required=True,\
        help="Provide the name of dataset to load an appropiate dataset to train.")
    parser.add_argument('--num_epochs', type=int, default=5, help="Enter the number of epochs to train the network.")
    parser.add_argument('--loss_type', type=str, default='Bce', choices=['Bce', 'Crossentropy'],\
         help="Enter the desired loss type.")
    parser.add_argument('--scheduler_type', type=str, default='LambdaLR', choices=['LambdaLR', 'MultiplicativeLR', 'StepLR', 'MultiStepLR'\
        'ExponentialLR', 'ReduceLROnPlateau', 'CyclicLR'], help="Enter the scheduler type to vary the learning rate.")
    parser.add_argument('--log_interval', type=int, default=100, help="Enter the intervals after which to log the details.")
    parser.add_argument('--num_classes', type=int, default=2, help="Enter the number of classes for a given dataset.")
    parser.add_argument('--batch_size', type=int, default=64, help="Enter the number of batch size.")
    parser.add_argument('--optimizer', type=str, default='Adam', choices=['Adam', 'SGD', 'RAdam'],\
        help="Enter the optimizer required to step during training.")
    parser.add_argument('--lr', type=int, default=1e-3, help="Enter the learning rate to train the network.")
    parser.add_argument('--augmentation', type=int, choices=[0, 1], default=1,\
         help="Augmentations to the training set to make the network more generalizable.")
    parser.add_argument('--save_path', type=str, default='./saved/models',\
        help="Save path for models.")
    parser.add_argument('--logdir', type=str, default='./saved/log', help="Log dir for tensorboard.")
    parser.add_argument('--flush_history', type=int, choices=[0, 1], default=0, help="Flush the tensorboard log dir.")
    parser.add_argument('--load_model', type=int, choices=[0, 1], default=0, help="Start training the model from a checkpoint.")
    parser.add_argument('--parallel', type=int, choices=[0, 1], default=1, help="Activate if multiple gpus used.")
    parser.add_argument('--val', type=int, choices=[0, 1], default=1, help="Activate to use validation set.")
    parser.add_argument('--return_best', type=int, choices=[0, 1], default=1, help="Returns the best model after training.")

    return args



if __name__ == "__main__":
    args = parse_arguments()
    run(args)










    
            
        


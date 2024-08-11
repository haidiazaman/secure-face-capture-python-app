from eye_dataset_v1 import Eye_Dataset
from eye_model_v1 import Eye_Net
from losses import FocalLoss
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts 

import argparse,torch
from torch.utils.data import DataLoader
import torch.distributed as dist
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import random
from glob import glob
from alive_progress import alive_it
from alive_progress import config_handler
config_handler.set_global(length = 20, force_tty = True)
from pathlib import Path
from scipy.special import softmax
from torchsummary import summary
import yaml
import time
import torch.nn as nn
# from torch.utils.tensorboard import Writer
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

# to calc time taken per epoch and entire training time
def convert_seconds_to_hms(seconds):
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return int(hours), int(minutes), int(seconds)

# accuracy function
def get_accuracy(outputs,labels):
    predicted_classes = torch.argmax(nn.Softmax(dim=1)(outputs),dim=1).cpu().detach().numpy()
    correct_predictions=(predicted_classes==labels.cpu().detach().numpy()).sum().item()
    accuracy = correct_predictions/labels.size(0) * 100
    return accuracy

def get_val_acc_from_cm(cm):
    val_open_acc = round(cm[0,0]/sum(cm[0])*100,2)
    val_close_acc = round(cm[1,1]/sum(cm[1])*100,2)
    val_block_acc = round(cm[2,2]/sum(cm[2])*100,2)
    return val_open_acc, val_close_acc, val_block_acc
    
    
def train_model(args):
    reverse_mapping = {0: 'open', 1: 'close', 2: 'block'}    
    # load input params
    with Path(args.config).open() as f:
        args=yaml.safe_load(f)
    # input params
    input_data_folder = args['input_data_folder'] # csvpath, train val shud be in same csv
    output_folder = args['output_folder']  # (str): Folder name of output in ./result/
    model_key = args['model_key']
    balance = eval(args['balance']) # (bool): set to False if using Focal Loss
    preload = eval(args['preload'])
    load_model = args['load_model'] # (str): Load model in ./result/ for finetuning, set to None to train from scratch
    num_classes = args['num_classes']
    if load_model=='None':
        load_model=None
    num_workers = args['num_workers'] # (int): The number of workers
    seed = args['seed'] # (int): Fix random seed
    lr = args['lr'] # (float): Learning rate for training
    batch_size = args['batch_size']
    epochs = args['epochs']
    class_weights = eval(args['class_weights'])
    input_dim = args['input_dim']
    weight_decay = args['weight_decay']
    first_cycle_steps = args['first_cycle_steps']
    cycle_mult = args['cycle_mult']
    max_lr = args['max_lr']
    min_lr = args['min_lr']
    warmup_steps = args['warmup_steps']
    gamma = args['gamma']
    
    
    # epochs = 2
    
    
    # initialise device
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')
    # initialization
    if seed: setup_seed(seed)
    # initialise output path
    output_path=Path(output_folder)
    output_path.mkdir(exist_ok = True, parents = True)
    
    weights_folder_path=os.path.join(output_path,'weights')
    weights_folder_path=Path(weights_folder_path)
    weights_folder_path.mkdir(exist_ok = True, parents = True)

    
    # setup model - from scratch or load model for finetune
    model = Eye_Net(
        model_key = model_key,
        in_channel = 3
    ).to(device)

    if load_model:
        model.load_state_dict(torch.load(os.path.join('./result', load_model), map_location = device))
        _ = model.eval()
    if device==torch.device('cuda'):
        model = nn.DataParallel(model)

    # setup optimizer and LR scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr = lr, weight_decay = weight_decay)
    scheduler = CosineAnnealingWarmupRestarts(optimizer,
                                              first_cycle_steps=first_cycle_steps,
                                              cycle_mult=cycle_mult,
                                              max_lr=max_lr,
                                              min_lr=min_lr,
                                              warmup_steps=warmup_steps,
                                              gamma=gamma)
    # setup loss function - focal loss + no balance or crossentropyloss + balance
    class_weights = torch.tensor(class_weights, device=device) # change to 1-%of that class
    criterion = FocalLoss(alpha=class_weights).to(device)

    # setup dataset and dataloader
    train_dataset = Eye_Dataset(input_data_folder, data_type = 'train', input_dim = input_dim,
                                seed = seed, balance = balance, augmentation = True, preload = preload)
    val_dataset = Eye_Dataset(input_data_folder, data_type = 'val', input_dim = input_dim,
                              seed = seed, balance = balance, augmentation = False, preload = preload)
    train_data_loader = DataLoader(train_dataset, num_workers = num_workers, batch_size = batch_size, shuffle = True, pin_memory = True)
    val_data_loader = DataLoader(val_dataset, num_workers = num_workers, batch_size = batch_size, shuffle = False, pin_memory = True)
                            
    # setup output csv to store train and val ave loss and acc for each epoch
    output_dict = {
        'epoch': ['' for epoch in range(epochs)],
        'lr': [0. for epoch in range(epochs)],
        'train_loss': [0. for epoch in range(epochs)],
        'val_loss': [0. for epoch in range(epochs)],
        'train_acc': [0. for epoch in range(epochs)],
        'val_acc': [0. for epoch in range(epochs)],
        'train_time_taken': [0. for epoch in range(epochs)],
        'val_time_taken': [0. for epoch in range(epochs)],
        'val_conf_mat': [[] for epoch in range(epochs)],
    }
    output_df=pd.DataFrame.from_dict(output_dict)
    output_df_path=os.path.join(output_path,f'results.csv')

    torch.autograd.set_detect_anomaly(True)
    print('######################')
    print('######################')
    print('### START TRAINING ###')
    print()

    training_start_time=time.time()
    best_val_loss = float('inf')  # Initialize with positive infinity or any large value
    best_val_acc = float('-inf')  # Initialize with positive infinity or any large value
    for epoch in range(epochs):
        #training loop
        model.train()
        train_running_loss,train_running_acc,train_start_time=0.,0.,time.time()
        for ind,(images,labels) in enumerate(alive_it(train_data_loader)):
            images=images.to(device)
            labels=labels.to(device)
            outputs=model(images)
            train_loss=criterion(outputs,labels)
            train_acc=get_accuracy(outputs,labels)
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            train_running_loss+=train_loss.item()
            train_running_acc+=train_acc
        train_time_taken=time.time()-train_start_time
        epoch_ave_train_loss=train_running_loss/(ind+1)
        epoch_ave_train_acc=train_running_acc/(ind+1)

        #validation loop
        model.eval()
        val_running_loss,val_running_acc,val_start_time=0.,0.,time.time()
        val_conf_matrix = np.zeros((num_classes,num_classes))
        for ind,(images,labels) in enumerate(alive_it(val_data_loader)):
            images=images.to(device)
            labels=labels.to(device)
            with torch.no_grad():
                outputs=model(images)
                val_loss=criterion(outputs,labels)
                val_acc=get_accuracy(outputs,labels)
                val_running_loss+=val_loss.item()
                val_running_acc+=val_acc
                y_pred = torch.argmax(nn.Softmax(dim=1)(outputs),dim=1).cpu().detach().numpy()
                y_label = labels.cpu().detach().numpy()
                
                for i,j in zip(y_label,y_pred):
                    val_conf_matrix[i][j]+=1
                    
        val_time_taken=time.time()-val_start_time
        epoch_ave_val_loss=val_running_loss/(ind+1)
        epoch_ave_val_acc=val_running_acc/(ind+1)

        val_conf_matrix=val_conf_matrix.astype(int)
        # val_acc_per_class_dict={}
        # for i in range(num_classes):
        #     class_name=mapping[i]
        #     class_acc=val_conf_matrix[i][i]/val_conf_matrix[i].sum()*100
        #     val_acc_per_class_dict[class_name]=class_acc    
        # output_per_class_acc_string = ', '.join([f'Acc-{class_name}: {round(acc,2)}%' for class_name, acc in val_acc_per_class_dict.items()])

        print()
        print(f'Epoch {epoch}, Training loss: {epoch_ave_train_loss:.4f}, Training acc: {epoch_ave_train_acc:.4f}, Validation loss: {epoch_ave_val_loss:.4f}, Validation acc: {epoch_ave_val_acc:.4f}')
        # print(f'Validation {output_per_class_acc_string}')
        print(f'Validation confusion matrix:\n{val_conf_matrix}')
        for c in range(len(val_conf_matrix)):
            acc_ = round(val_conf_matrix[c,c]/sum(val_conf_matrix[c]) *100 , 2)
            print(f'val_acc-{reverse_mapping[c]}: {acc_}%')
            
        print()

        #save values to df
        output_df.loc[epoch,'epoch']=epoch
        output_df.loc[epoch,'lr']=scheduler.get_lr()[0]
        output_df.loc[epoch,'train_loss']=round(epoch_ave_train_loss,4)
        output_df.loc[epoch,'val_loss']=round(epoch_ave_val_loss,4)
        output_df.loc[epoch,'train_acc']=round(epoch_ave_train_acc,2)
        output_df.loc[epoch,'val_acc']=round(epoch_ave_val_acc,2)
        output_df.loc[epoch,'train_time_taken']=round(train_time_taken,2)
        output_df.loc[epoch,'val_time_taken']=round(val_time_taken,2)
        # output_df.loc[epoch,'val_acc_real']=round(val_acc_per_class_dict['real'],2)
        # output_df.loc[epoch,'val_acc_deepfake']=round(val_acc_per_class_dict['deepfake'],2)
        for x in val_conf_matrix.ravel().tolist():
            output_df.loc[epoch,'val_conf_mat'].append(x) 
            
        output_df.to_csv(output_df_path,index=False)


#         # TENSORBOARD - plot graphs
#         writer.add_scalar('train/training_loss',epoch_ave_train_loss,epoch)
#         writer.add_scalar('val/validation_loss',epoch_ave_val_loss,epoch)
#         writer.add_scalar('train/training_acc',epoch_ave_train_acc,epoch)
#         writer.add_scalar('val/validation_acc',epoch_ave_val_acc,epoch)
#         writer.add_scalar('learning_rate_scheduler',scheduler.get_lr()[0],epoch)

        # # Save the model if the current validation loss is the best so far - this will only save the model not overwrite previous
        # if epoch_ave_val_loss < best_val_loss or epoch_ave_val_acc > best_val_acc:
        #     best_val_loss = epoch_ave_val_loss
        #     best_val_acc = epoch_ave_val_acc
        #     if device == torch.device('cpu'): # for training on CPU, model doesnt not nn.DataParallel, so saving dont need the module in model.module.state_dict()
        #         torch.save(model.state_dict(), os.path.join(weights_folder_path,f'epoch{str(epoch)}.pt'))
        #     else:
        #         torch.save(model.module.state_dict(), os.path.join(weights_folder_path,f'epoch{str(epoch)}.pt'))
        
        
        # save the model after every epoch
        if device == torch.device('cpu'): # for training on CPU, model doesnt not nn.DataParallel, so saving dont need the module in model.module.state_dict()
            torch.save(model.state_dict(), os.path.join(weights_folder_path,f'epoch{str(epoch)}.pt'))
        else:
            torch.save(model.module.state_dict(), os.path.join(weights_folder_path,f'epoch{str(epoch)}.pt'))
                
        # print('ADD IN CODE SNIPPET TO SAVE ONLY 5 BEST MODELS VAL LOSS')
            
        scheduler.step()        
        
    print()
    print()
    print('### COMPLETED TRAINING ###')
    total_time_taken=time.time()-training_start_time
    h,m,s=convert_seconds_to_hms(total_time_taken)
    print(f'total time taken: {h}h, {m}m, {s}s')
    # writer.close()

    
if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Pass input params to separate config file in .json format and then call it using parser")
    parser.add_argument("--config", type=str, help="Path to JSON config file.")
    args = parser.parse_args()
    train_model(args)
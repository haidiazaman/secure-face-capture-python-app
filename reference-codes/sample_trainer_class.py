import torch
import time
import os
import datetime
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
from math import sqrt
from utils.util import format_time
from utils.optimizer import get_optimizer
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr, spearmanr
from utils.loss import HardConLoss
# %matplotlib inline
import matplotlib.pyplot as plt




class Trainer(nn.Module):
    def __init__(self, device, model, epochs, data_group, batch_size, train_dl, test_dl, dev_dl=None, model_dir='/models/', logging_step=500, beta=1, save_point=0.85, save_per_epoch=False, lr_scale=100, lr=1e-05, temperature=0.05, save_last=True, seed=42):
        super(Trainer, self).__init__()
        self.device = device
        self.model = model
        self.epochs = epochs
        self.lr = lr
        self.lr_scale = lr_scale
        self.data_group = data_group
        self.batch_size = batch_size
        self.model_dir = model_dir
        self.save_point = save_point
        self.logging_step = logging_step
        self.save_per_epoch = save_per_epoch
        self.optimizer = get_optimizer(self.model, lr=lr, lr_scale=lr_scale)
        self.beta = beta
        self.seed_val = seed
        self.train_dataloader = train_dl
        self.valid_dataloader = dev_dl
        self.test_dataloader = test_dl
        self.total_steps = len(self.train_dataloader) * epochs
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=0, num_training_steps=self.total_steps)
        self.inst_disc_loss = HardConLoss(temperature=temperature, device=device).to(device)
        self.mse_loss = nn.MSELoss().to(device)

        # Track best validation loss and Pearson correlation
        self.best_valid_loss = float('inf')
        self.best_pearson = -1.0
        self.best_spearman = -1.0
        self.gstep = 0
        self.ckpt = 0

    def train(self):
        print('\n======================Training Started======================')
        self.model.train()
        training_stats = []
        total_t0 = time.time()

        for epoch in tqdm(range(self.epochs), desc="Training "):
            t0 = time.time()
            total_train_loss = 0

            for train_data in tqdm(self.train_dataloader, desc=f'Epoch ({epoch+1} / {self.epochs}) '):
                loss = self.train_step(train_data)
                total_train_loss += loss.item()

                # Validation Checkpoint
                if (self.gstep>0) and (self.gstep%self.logging_step==0):
                    if self.valid_dataloader:
                        self.validate()
                self.gstep += 1

            avg_train_loss = total_train_loss / len(self.train_dataloader)
            training_time = format_time(time.time() - t0)

            # Validation
            if self.valid_dataloader:
                avg_valid_loss, pearson, spearman, rmse = self.validate()

                print(f"Average training loss: {avg_train_loss:.5f}")
                print(f"Training epoch took: {training_time}")

                training_stats.append({
                    'epoch': epoch + 1,
                    'Train Loss': avg_train_loss,
                    'Train Time': training_time,
                    'Val Loss': avg_valid_loss,
                    'Val R': pearson,
                    'Val Rho': spearman,
                    'Val RMSE': rmse
                })

                # Check if we should save the model
                if (pearson > self.best_pearson or spearman > self.best_spearman) and pearson >= self.save_point and self.save_per_epoch:
                    self.best_valid_loss = avg_valid_loss
                    self.best_pearson = pearson
                    self.best_spearman = spearman
                    self.ckpt += 1
                    self.save_model(
                        epoch=epoch,
                        loss=avg_train_loss,
                        pearson=pearson,
                        spearman=spearman,
                        ckpt=self.ckpt
                    )
            else:
                print(f"Average training loss: {avg_train_loss:.5f}")
                print(f"Training epoch took: {training_time}")

                training_stats.append({
                    'epoch': epoch + 1,
                    'Train Loss': avg_train_loss,
                    'Train Time': training_time,
                })
        self.save_model(name='last_trained.pt')
        print("\n=====================Training Completed=====================")
        print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))

        df_stats = pd.DataFrame(data=training_stats)
        df_stats = df_stats.set_index('epoch')
        print(df_stats)
        # self.plot_curves(training_stats)
        return

    def train_step(self, train_data):
        self.model.zero_grad()
        scores = train_data['score'].to(self.device)
        preds, feat_1, feat_2 = self._common_step(data=train_data)

        loss = self.mse_loss(preds, scores.squeeze().to(self.device))
        pair_loss = (self.beta * self.inst_disc_loss(feat_1.squeeze(0), feat_2.squeeze(0), scores)).float()
        loss +=  pair_loss

        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.scheduler.step()
        return loss

    def validate(self):
        self.model.eval()
        total_valid_loss = 0
        predictions = []
        true_labels = []

        for valid_data in tqdm(self.valid_dataloader, desc="Validation Running "):
            scores = valid_data['score'].to(self.device)
            outputs, _, _ = self._common_step(data=valid_data, train=False)
            loss = self.mse_loss(outputs, scores.squeeze().to(self.device))
            total_valid_loss += loss.item()

            predictions.extend(outputs.cpu().numpy())
            true_labels.extend(scores.cpu().numpy())

        self.model.train()
        avg_valid_loss = total_valid_loss / len(self.valid_dataloader)
        pearson = pearsonr(predictions, true_labels)[0]
        spearman = spearmanr(predictions, true_labels)[0]
        rmse = sqrt(mean_squared_error(predictions, true_labels))

        print(f"Validation loss: {avg_valid_loss:.5f} \t r: {pearson:.4f} \t rho: {spearman:.4f} \t rmse: {rmse:.4f}")
        return avg_valid_loss, pearson, spearman, rmse

    def test(self):
        predictions = []
        true_labels = []
        self.model.eval()

        print("\n=====================Evaluation Started=====================")

        for test_data in tqdm(self.test_dataloader, desc="Evaluating "):
            scores = test_data['score'].to(self.device)
            outputs, _, _ = self._common_step(data=test_data, train=False)

            predictions.extend(outputs.cpu())
            true_labels.extend(scores.cpu())

        # Compute pearson and spearman correlation coefficient and rmse
        r = pearsonr(predictions, true_labels)
        rho = spearmanr(predictions, true_labels)
        rmse = sqrt(mean_squared_error(predictions, true_labels))

        print("\n====================Evaluation Completed====================")
        print("Pearson: {:}".format(round(r.statistic, 4)))
        print("Spearman: {:}".format(round(rho.statistic, 4)))
        print("RMSE: {:}".format(round(rmse,4)))
        print("============================================================\n")
        return

    def _common_step(self, data, train=True):
        input_ids = data['feats']['input_ids'].to(self.device)
        attention_masks = data['feats']['attention_mask'].to(self.device)

        features = [{'input_ids': input_id, 'attention_mask': attention_mask}
                    for input_id, attention_mask in zip(input_ids, attention_masks)]

        if train==True:
            predictions, feat_1, feat_2 = self.model(features)
        else:
            with torch.no_grad():
                predictions, feat_1, feat_2 = self.model(features)
        return predictions, feat_1, feat_2

    def save_model(self, epoch=0, loss=0, pearson=0.0, spearman=0.0, name=None, ckpt=0):
        print("Model saving...")
        path = self.generate_path(name=name, ckpt=ckpt, pearson=pearson, spearman=spearman)
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'epoch': epoch,
            'loss': loss
        }
        torch.save(checkpoint, path)
        print(f'Model saved to path: {path}')
        return

    def load_model(self, name):
        print("Model loading...")
        path = self.model_dir + name
        checkpoint = torch.load(path)

        model = self.model
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer = get_optimizer(model, lr=self.lr, lr_scale=self.lr_scale)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=self.total_steps)
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

        print('Model loaded!')
        return checkpoint['epoch'], checkpoint['loss']

    def generate_path(self, pearson=0.0, spearman=0.0, name=None, ckpt=0):
        # os.makedirs(os.path.dirname(self.model_dir), exist_ok=True)
        if name==None:
            model_path = f'{self.model_dir}P{pearson:.4f}.S{spearman:.4f}_DG{self.data_group}_E{self.epochs}_B{self.batch_size}_LR{self.lr}_ckpt{ckpt}.pt'
        else:
            model_path = f'{self.model_dir}{name}'
        return model_path

    def plot_curves(self, training_stats):
        epochs = [stat['epoch'] for stat in training_stats]
        train_losses = [stat['Train Loss'] for stat in training_stats]
        val_losses = [stat['Val Loss'] for stat in training_stats]
        val_r = [stat['Val R'] for stat in training_stats]
        val_rho = [stat['Val Rho'] for stat in training_stats]

        # Create figure and axes
        fig, axs = plt.subplots(2, 1, figsize=(10, 12))

        # Plotting training and validation loss curves
        axs[0].plot(epochs, train_losses, label='Train Loss', marker='o')
        axs[0].plot(epochs, val_losses, label='Val Loss', marker='s')
        axs[0].set_xlabel('Epoch')
        axs[0].set_ylabel('Loss')
        axs[0].set_title('Training and Validation Loss Curves')
        axs[0].legend()
        axs[0].grid(True)

        # Plotting validation Pearson's R and Spearman's Rho curves
        axs[1].plot(epochs, val_r, label='Val R (Pearson)', marker='o')
        axs[1].plot(epochs, val_rho, label='Val Rho (Spearman)', marker='s')
        axs[1].set_xlabel('Epoch')
        axs[1].set_ylabel('Correlation Coefficient')
        axs[1].set_title('Validation Pearson\'s R and Spearman\'s Rho Curves')
        axs[1].legend()
        axs[1].grid(True)

        # Adjust layout to prevent overlap
        plt.tight_layout()

        # Display the plot
        plt.show()
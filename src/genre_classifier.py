import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os

from data_preparation_GTZAN import fix_length, divide_into_segments


class GenreClassifier(nn.Module):
    def __init__(self, device=None):
        super().__init__()
        self.device = device if device else torch.device('cpu')
        self.conv_pool_stack = nn.Sequential(
                                                nn.Conv2d(1, 32, 3, padding=1), 
                                                nn.BatchNorm2d(32),
                                                nn.ReLU(),
                                                nn.Conv2d(32, 32, 3, padding=1, stride=2),  # Added convolution instead of pooling
                                                nn.BatchNorm2d(32),
                                                nn.ReLU(),
                                                nn.MaxPool2d(3, stride=2),
                                                nn.Dropout2d(0.3),

                                                nn.Conv2d(32, 64, 3, padding=1),
                                                nn.BatchNorm2d(64),
                                                nn.ReLU(),
                                                nn.Conv2d(64, 64, 3, padding=1, stride=2),
                                                nn.BatchNorm2d(64),
                                                nn.ReLU(),
                                                nn.MaxPool2d(3, stride=2),
                                                nn.Dropout2d(0.3),

                                                nn.Conv2d(64, 128, 3, padding=1),
                                                nn.BatchNorm2d(128),
                                                nn.ReLU(),
                                                nn.MaxPool2d(3, stride=2))
        self.flatten = nn.Flatten()
        self.fully_connected = nn.Sequential(
                                        nn.Dropout(0.3),
                                        nn.Linear(1152, 512),
                                        nn.ReLU(),
                                        nn.Dropout(0.5),
                                        nn.Linear(512, 10))
        self.history = {'train_loss' :[],
                        'val_loss': [],
                        'train_acc': [],
                        'val_acc' : []}
        
        
    def forward(self, spec, targets=None):
        x = self.conv_pool_stack(spec)
        x = self.flatten(x)
        logits = self.fully_connected(x)
        
        if targets is None:
            loss = None
        else:
            loss = F.cross_entropy(logits, targets, label_smoothing=0.1)
            
        return logits, loss
    
    
    def evaluate(self, dataloader, majority_vote=None, eval_iters=100):
        self.eval()
        total_loss = 0
        total_samples = 0
        correct = 0
        
        track_predictions = {}
        track_labels = {}
        i = 0
        with torch.no_grad():
            for batch in dataloader:
                if i == eval_iters:
                    break
                i += 1
                inputs, labels, track_ids = batch
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                logits, loss = self(inputs, labels)
                total_loss += loss.item() * inputs.shape[0]
                total_samples += inputs.shape[0]
                
                preds = logits.argmax(dim=1)
                
                if majority_vote is None:
                    correct += (preds==labels).sum().item()
                else:
                    for track_id, pred, logit, label in zip(track_ids, preds, logits, labels):
                        if track_id not in track_predictions:
                            track_predictions[track_id] = []
                            track_labels[track_id] = label.item()

                        if majority_vote == 'hard':
                            track_predictions[track_id].append(pred.item())
                        elif majority_vote == 'soft':
                            track_predictions[track_id].append(logit.cpu().numpy())
                    
        if majority_vote is None:
            accuracy = correct / total_samples
        else:
            correct_tracks = 0
            for track_id in track_predictions:
                if majority_vote == 'hard':
                    final_pred = max(set(track_predictions[track_id]), key=track_predictions[track_id].count)
                elif majority_vote == 'soft':
                    logits_list = np.array(track_predictions[track_id])
                    avg_logits = np.mean(logits_list, axis=0)
                    final_pred = np.argmax(avg_logits)

                if final_pred == track_labels[track_id]:
                    correct_tracks += 1

            accuracy = correct_tracks / len(track_predictions)

        avg_loss = total_loss / total_samples

        self.train()
        return avg_loss, accuracy
    
    
    def fit(self, train_loader, val_loader, optimizer, max_epochs, eval_iters=100, scheduler=None):
        self.train()
        for i in range(max_epochs):
            for batch in train_loader:
                inputs, labels, _ = batch
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                logits, loss = self(inputs, labels)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                if scheduler is not None:
                    if isinstance(scheduler, torch.optim.lr_scheduler.CyclicLR):
                        scheduler.step()
            train_loss, train_acc = self.evaluate(train_loader, None, eval_iters)
            val_loss, val_acc = self.evaluate(val_loader, None, eval_iters)
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss)
                elif not isinstance(scheduler, torch.optim.lr_scheduler.CyclicLR):
                    scheduler.step()
            print(f'Epoch [{i+1}/{max_epochs}] | Train loss : {train_loss:.4f} | Train acc : {train_acc:.2f} | Val loss : {val_loss:.4f} | Val acc : {val_acc:.2f}')
            

    def predict(self, dataloader, majority_vote=None):
         self.eval()
         with torch.no_grad():   
            predictions = []
            true_labels = []
            track_predictions = {}
            track_labels = {}

            for batch in dataloader:
                inputs, labels, track_ids = batch
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                logits, _ = self(inputs)
                preds = logits.argmax(dim=-1).cpu().numpy().tolist()
                if majority_vote is None:
                    predictions += preds
                    true_labels += labels.cpu().numpy().tolist()
                else:
                    for track_id, pred, logit, label in zip(track_ids, preds, logits, labels):
                        if track_id not in track_labels:
                            track_predictions[track_id] = ([], label.item())
                        if majority_vote == 'hard':
                            track_predictions[track_id][0].append(pred)
                        else:
                            track_predictions[track_id][0].append(logit.cpu().numpy())                
                
            if majority_vote is None:
                self.train()
                return predictions, true_labels
            else:
                final_predictions = []
                for track_id in track_predictions:
                        if majority_vote == 'hard':
                            votes = track_predictions[track_id][0]
                            final_pred = max(set(votes), key=votes.count)
                        elif majority_vote == 'soft':
                            logits_list = np.array(track_predictions[track_id][0])
                            avg_logits = np.mean(logits_list, axis=0)
                            final_pred = np.argmax(avg_logits)
                        final_predictions.append(final_pred)
                        true_labels.append(track_predictions[track_id][1])
                self.train()
                return final_predictions, true_labels
            

    def inference(self, raw_audio, sr):
        self.eval()
        with torch.no_grad():
            y = fix_length(raw_audio, sr)
            spectrograms = divide_into_segments(y, save=False)
            spectrograms = [torch.tensor(spectrogram).unsqueeze(0) for spectrogram in spectrograms]
            spectrograms = torch.stack(spectrograms).to(self.device)

            logits, _ = self(spectrograms)
            avg_logits = logits.mean(dim=0)
            pred = avg_logits.argmax(dim=0).item()
            probs = F.softmax(avg_logits, dim=0)
        self.train()
        return pred, probs


    def plot_training_curves(self, save=False, save_dir='plots'):
        if save:
            os.makedirs(save_dir, exist_ok=True)
        epochs = np.arange(1, len(self.history['train_loss']) + 1)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(epochs, self.history['train_loss'], label='Training Loss', color='tab:blue', linewidth=2)
        ax.plot(epochs, self.history['val_loss'], label='Validation Loss', color='tab:orange', linewidth=2)

        ax.set_xlabel('Epochs', fontsize=14)
        ax.set_ylabel('Loss', fontsize=14)
        ax.set_title('Training vs Validation Loss', fontsize=16)
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.xaxis.set_major_locator(plt.MaxNLocator(10))
        for label in ax.get_xticklabels():
            label.set_rotation(45)
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.legend(fontsize=12)
        plt.tight_layout()
        if save:
            plt.savefig(f'{save_dir}/loss_curve.png')
        plt.show()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(epochs, self.history['train_acc'], label='Training Acc', color='tab:blue', linewidth=2)
        ax.plot(epochs, self.history['val_acc'], label='Validation Acc', color='tab:orange', linewidth=2)

        ax.set_xlabel('Epochs', fontsize=14)
        ax.set_ylabel('Acc', fontsize=14)
        ax.set_title('Training vs Validation Acc', fontsize=16)
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.xaxis.set_major_locator(plt.MaxNLocator(10))
        for label in ax.get_xticklabels():
            label.set_rotation(45)
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.legend(fontsize=12)
        plt.tight_layout()
        if save:
            plt.savefig(f'{save_dir}/acc_curve.png')
        plt.show()


    def save(self, path):
        torch.save({'model_state': self.state_dict(),
                     'history': self.history}, 
                     path)


    def load(self, path):
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model_state'])
        self.history = checkpoint['history']

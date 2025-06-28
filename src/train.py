import json
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from spectrogram_augment import SpectrogramAugment
from spectrogram_dataset import SpectrogramDataset
from genre_classifier import GenreClassifier


def plot_confusion_matrix(y_true, y_pred, class_names, normalize=False, label=''):
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names, cbar=False)

    plt.ylabel('True Label', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.title('Confusion Matrix' + (' (Normalized)' if normalize else '') + label, fontsize=16)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()


def main():
    GPU = False
    device = torch.device('cuda' if (torch.cuda.is_available() and GPU) else 'cpu')
    print(f'Using device: {device}')

    train_csv = pd.read_csv('data/train.csv')
    val_csv = pd.read_csv('data/val.csv')
    test_csv = pd.read_csv('data/test.csv')
    with open('data/genre_id_to_name.json', 'r', encoding='utf-8') as f:
        genre_id_to_name = json.load(f)
    class_names = list(genre_id_to_name.values()) 

    data_train = train_csv['segment_paths'].to_numpy()
    labels_train = train_csv['genre_id'].to_numpy()

    data_val = val_csv['segment_paths'].to_numpy()
    labels_val = val_csv['genre_id'].to_numpy()

    data_test = test_csv['segment_paths'].to_numpy()
    labels_test = test_csv['genre_id'].to_numpy()

    dataset_train = SpectrogramDataset(labels_train,
                                        data_train,
                                        transform=SpectrogramAugment(time_mask_param=50, freq_mask_param=40, noise_level=1, shift_max=40))
    loader_train = DataLoader(dataset_train, batch_size=64, shuffle=True)

    dataset_val = SpectrogramDataset(labels_val, data_val)
    loader_val = DataLoader(dataset_val, batch_size=64, shuffle=False)

    dataset_test = SpectrogramDataset(labels_test, data_test)
    loader_test = DataLoader(dataset_test, batch_size=64, shuffle=False)

    model = GenreClassifier(device=device).to(device)
    model = torch.compile(model)
    optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=5e-2)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                    mode='min',
                                                    patience=3, 
                                                    factor=0.2)

    model.fit(loader_train, loader_val, optimizer, max_epochs=60, scheduler= scheduler)
    model.save('genre_classifier.pth')

    model.plot_training_curves(save=True, save_dir='plots')

    test_loss, test_acc = model.evaluate(loader_test, 'soft', 1000)
    print(f'Accuracy on test set with soft voting : {test_acc:.2f}')
    pred, true = model.predict(loader_test, 'soft')
    plot_confusion_matrix(true, pred, class_names,True, '[soft voting]')

    test_loss, test_acc = model.evaluate(loader_test, 'hard', 1000)
    print(f'Accuracy on test set with hard voting : {test_acc:.2f}')
    pred, true = model.predict(loader_test, 'hard')
    plot_confusion_matrix(true, pred, class_names, True, '[hard voting]')

    test_loss, test_acc = model.evaluate(loader_test, None, 1000)
    print(f'Accuracy on test set per sample : {test_acc:.2f}')
    pred, true = model.predict(loader_test, 'hard')
    plot_confusion_matrix(true, pred, class_names, True, '[per_sample]')


if __name__ == "__main__":
    main()
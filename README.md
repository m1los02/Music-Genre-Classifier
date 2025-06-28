# Music-Genre-Classifier

## Overview
This is a project for educational purposes, where I intended to expand my knowledge of convolutional neural networks (CNN). The goal of this project is to predict the genre of a 30-second music track based on its mel spectrogram. The model is trained on the GTZAN music dataset and evaluated using accuracy metrics. The final model achieves 85% accuracy on the test set using majority vote (hard voting).

The project is done in Python, using librosa for audio processing and the PyTorch framework for building the CNN model.

In the data directory you can find custom .csv files which connect tracks with paths to their segments and their genres, and a .json file which saves the mappings from genre names to their labels (all files are generated in the code).

The src directory has the code needed for the audio file preprocessing (data_preparation_GTZAN.py and prepare_dataset_GTZAN.py) and files needed for model definition, data augmentation, and dataloaders (genre_classifier.py, spectrogram_augment.py, spectrogram_dataset.py). By running train.py, the model is trained and saved.

## Dataset
I used the GTZAN dataset, which contains 10 genres and 1000 audio tracks. I converted the 30-second tracks into 3-second segments and then into mel spectrograms. The spectrograms were saved as .npy files and used afterward (with data augmentation) to train the model.

## Model
The model is a simple CNN with multiple convolutional layers, batch normalization, dropout, and max pooling. I also experimented with adding regularization (weight decay) and used a learning rate scheduler.

## Results
My best model achieved around 76% per-segment accuracy on the validation and test sets. On the test set, the model achieved around 85% accuracy per track (predicting genre for the full 30-second track based on majority voting). I also plotted a confusion matrix to see which genres were confused the most.

from collections import OrderedDict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing

import torch
import torch.nn.functional as F
from torch import nn
from torch import optim
from torch.utils.data import Dataset, DataLoader

from torchvision import models

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from preprocessing import sample_from_patient, get_transformed_image, clean_the_data
from pytorch_utils import EarlyStopping

# create custom dataset class for dataloader to ingest


class DicomDataSet(Dataset):
    def __init__(self, paths, labels):
        self.paths = paths
        self.labels = labels

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        image = get_transformed_image(self.paths[idx])
        label = self.labels[idx]
        return image, label


if __name__ == "__main__":
    MODALITIES = ['CT', 'CTA', 'CTP', 'NCCT', 'PET']
    REGION_DICT = {'Head': 'Brain',
                   'Brain': 'Brain',
                   'Chest': 'Chest',
                   'Abdomen': 'Abdomen'}
    NO_SAMPLES_PER_PATIENT = 3
    BATCH_SIZE = 5

    meta_df = pd.read_pickle('metadata.pkl')
    meta_df = clean_the_data(meta_df, MODALITIES, REGION_DICT)
    sampled_meta_df = sample_from_patient(meta_df, NO_SAMPLES_PER_PATIENT)
    paths = sampled_meta_df.Path.tolist()
    labels = sampled_meta_df.Class
    numm_classes = labels.Class.nunique()

    X_train, X_test, y_train, y_test = train_test_split(
        paths, labels, test_size=0.2, random_state=42)
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42)

    enc = LabelEncoder()
    y_train = enc.fit_transform(y_train)
    y_valid = enc.transform(y_valid)
    y_test = enc.transform(y_test)

    train_db = DicomDataSet(X_train, y_train)
    valid_db = DicomDataSet(X_valid, y_valid)
    test_db = DicomDataSet(X_test, y_test)
    trainloader = DataLoader(train_db, batch_size=BATCH_SIZE, shuffle=True)
    validloader = DataLoader(valid_db, batch_size=BATCH_SIZE, shuffle=True)
    testloader = DataLoader(test_db, batch_size=BATCH_SIZE, shuffle=True)
    # images, labels = next(iter(trainloader))
    # print(images.shape, labels.shape)
    # for image in images:
    #     plt.imshow(image.permute(1, 2, 0), cmap=plt.cm.gray)
    #     plt.show()

    # Load a pretrained model for transfer learning
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = models.resnet50(pretrained=True)

    # Freeze parameters so we don't backpropagate through them
    for param in model.parameters():
        param.requires_grad = False

    # The last layer is named fc and has 2048 input nodes and 1000 output nodes (for 1000 classes).
    # We'll replace it with the number of classes in our dataset

    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(2048, 500)),
        ('relu', nn.ReLU()),
        ('output', nn.Linear(500, numm_classes))
    ]))
    model.fc = classifier
    model.to(device)

    # combines LogSoftmax and NLLLoss in one single class
    criterion = nn.CrossEntropyLoss()

    # Train the model
    optimizer = optim.Adam(model.fc.parameters(), lr=0.003)

    epochs = 40
    patience = 3
    best_result = 1000
    train_losses, valid_losses = [], []
    batch_losses = []
    es = EarlyStopping(patience, best_result)
    for epoch in range(epochs):
        # training pass
        training_loss = 0
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            # CrossEntropyLoss requires its target to be a float tensor
            loss = criterion(logits, labels.type(torch.LongTensor).to(device))
            loss.backward()
            optimizer.step()
            training_loss += loss.item()
        else:
            # validation pass
            model.eval()
            with torch.no_grad():  # turn off gradients
                accuracy = 0
                validation_loss = 0
                for images, labels in validloader:
                    images, labels = images.to(device), labels.to(device)
                    logits = model(images)
                    loss = criterion(logits, labels.type(
                        torch.LongTensor).to(device))
                    validation_loss += loss.item()
                    predictions = torch.sigmoid(logits)
    #                 to do: calculate accuracy
    #                 accuracy += batch_accuracy
    #             accuracy /= len(validloader)
        # Get everage loss per training batch
        training_loss /= len(trainloader)
        train_losses.append(training_loss)
        # Get everage loss per validation batch
        validation_loss /= len(validloader)
        valid_losses.append(validation_loss)
        print(f'Epoch: {epoch+1}/{epochs}')
        print(f'Training_loss: {training_loss}')
        print(f'Validation_loss: {validation_loss}')
        # print(f'Accuracy: {accuracy.item()*100}%')
        # early stopping
        if es.stop(training_loss, validation_loss, epoch):
            break
        model.train()

    plt.plot(train_losses, label='train')
    plt.plot(valid_losses, label='valid')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

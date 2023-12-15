import os
from PIL import Image
from torch import optim
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch
import torch.nn as nn


class CustomImageDataset(Dataset):
    def __init__(self, root_folder_images, root_folder_labels, file_names, transform=None):
        self.root_folder_images = root_folder_images
        self.root_folder_labels = root_folder_labels
        self.file_names = file_names
        self.transform = transform

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, index):
        file_name = self.file_names[index]

        # Percorsi completi alle immagini e alle etichette
        image_path = os.path.join(self.root_folder_images, file_name + ".jpg")
        label_path = os.path.join(self.root_folder_labels, file_name + ".csv")

        # Carica l'immagine
        image = Image.open(image_path).convert("RGB")

        # Carica le etichette dal file (questo è solo un esempio, adatta in base al tuo formato)
        labels = self._load_labels(label_path)

        # Applica le trasformazioni all'immagine se specificato
        if self.transform:
            image = self.transform(image)

        return {'image': image, 'labels': labels}

    def _load_labels(self, label_path):
        labels = []
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                bbox = {
                    'Left': int(parts[0]),
                    'Top': int(parts[1]),
                    'Right': int(parts[2]),
                    'Bottom': int(parts[3]),
                    'Label_ID': int(parts[4]),
                    'Stem_X': int(parts[5]),
                    'Stem_Y': int(parts[6]),
                }
                labels.append(bbox)
        return labels

class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()

        # Input shape is 256x256x3
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)

        # Output shape is 128x128x32
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)

        # Output shape is 64x64x64
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=True)

        # Output shape is 32x32x128
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.relu4 = nn.ReLU(inplace=True)

        # Output shape is 16x16x256
        self.conv5 = nn.Conv2d(256, 512, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        self.relu5 = nn.ReLU(inplace=True)

        # Output shape is 8x8x512
        self.conv6 = nn.Conv2d(512, 1024, 3, padding=1)
        self.bn6 = nn.BatchNorm2d(1024)
        self.relu6 = nn.ReLU(inplace=True)

        # Fully connected layer with 4096 units
        self.fc1 = nn.Linear(1024 * 8 * 8, 4096)
        self.bn7 = nn.BatchNorm1d(4096)
        self.relu7 = nn.ReLU(inplace=True)

        # Output layer with 2 classes (crop and weed)
        self.fc2 = nn.Linear(4096, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu5(x)

        x = self.conv6(x)
        x = self.bn6(x)
        x = self.relu6(x)

        x = x.view(-1, 1024 * 8 * 8)
        x = self.fc1(x)
        x = self.bn7(x)
        x = self.relu7(x)

        x = self.fc2(x)
        return x


if __name__ == '__main__':
    print("######################    START MAIN      ######################")
    # Esempio di utilizzo
    root_folder_images = 'data/images'
    root_folder_labels = 'data/bboxes/CropOrWeed2'

    with open("../test_split.txt", 'r') as file:
        test_list = [line.strip() for line in file.readlines()]

    with open("../train_split.txt", 'r') as file:
        train_list = [line.strip() for line in file.readlines()]

    transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
    print("######################    START DATASET      ######################")
    train_set = CustomImageDataset(root_folder_images, root_folder_labels, train_list, transform=transform)
    test_set = CustomImageDataset(root_folder_images, root_folder_labels, test_list, transform=transform)
    print("Loading completato")
    # Esempio di accesso a un campione
    sample = train_set[0]
    image = sample['image']
    labels = sample['labels']

    # Esempio di iterazione attraverso il dataset
    for i in range(3):
        sample = train_set[i]
        print(f"Sample {i + 1} - Image shape: {sample['image'].shape}, Number of labels: {len(sample['labels'])}")

    ################################################################
    print("######################    START TRAINING      ######################")
    # Training settings
    batch_size = 4
    num_epochs = 4
    lr = 0.001

    # Load the dataset and create data loader
    trainloader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2)
    model = CustomCNN()
    print("trainloader e model completato")
    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    print("inizio il training")
    # Train the model
    for epoch in range(num_epochs):
        for i, data in enumerate(trainloader):
            print('i')
            print(i)
            print('data')
            print(data)
            # Get the inputs
            inputs, labels = data

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Print the loss
            if (i + 1) % 100 == 0:
                print(f"Epoch {epoch + 1}, Step {i + 1}: Loss: {loss.item()}")

    # # Salvataggio del modello
    # torch.save(model.state_dict(), 'custom_cnn_model.pth')
    #
    # # Salvataggio dello stato dell'ottimizzatore se necessario
    # torch.save(optimizer.state_dict(), 'optimizer_state.pth')

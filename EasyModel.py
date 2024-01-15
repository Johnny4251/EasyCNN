import torch
from torch import nn 
from torchvision.transforms import v2
from torch.utils.data import DataLoader
from PIL import Image
import os



# Designed for self made datasets
# folder should be composed of /data/test & /data/train
# Uses adam optimizer and cross entropy loss

class EasyCNN:
    example_train_transform = v2.Compose([
        v2.ColorJitter(brightness=0.2, contrast=0.2),
        v2.RandomAdjustSharpness(sharpness_factor=1.5),
        v2.RandomHorizontalFlip(),
        v2.RandomRotation(degrees=25),
        v2.ToImageTensor(),
        v2.ConvertImageDtype()
    ])

    simple_transform = v2.Compose([
        v2.ToTensor(),
        v2.ConvertImageDtype()
    ])

    vgg_train_transform = v2.Compose([
        v2.Resize((224, 224)),
        v2.RandomHorizontalFlip(),
        v2.ToTensor(),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    vgg_test_transform = v2.Compose([
        v2.Resize((224, 224)),
        v2.ToTensor(),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    def __init__(self, model, train_transform=simple_transform, grayscale=False):
        self.train_transform = train_transform
        self.test_transform = self.simple_transform
        self.model = model
        self.grayscale = grayscale

        self.device = ("cuda" if torch.cuda.is_available() else "cpu")
        print("Using ", self.device)

    def __train_one_epoch(self, train_dataloader, loss_fn, optimizer):
        size = len(train_dataloader.dataset)
        self.model.train()

        for batch, (X,y) in enumerate(train_dataloader):

            X = X.to(self.device)
            y = y.to(self.device)
            
            pred = self.model(X)
            loss = loss_fn(pred, y)
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            if batch%100 == 0:
                loss = loss.item()
                index = (batch+1)*len(X)
                print(index, "of", size, ": Loss =", loss)

    def __test(self, dataloader, loss_fn):
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        self.model.eval()
        
        test_loss = 0
        correct = 0
        
        with torch.no_grad():
            for X,y in dataloader:
                X = X.to(self.device)
                y = y.to(self.device)
                pred = self.model(X)
                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            test_loss /= num_batches
            correct /= size
    
        print("\tAccuracy:", correct)
        print("\tLoss:", test_loss)

    def train_model(self, total_epochs, batch_size, save_model=False, model_name="DEFAULT"):
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

        self.model.to(self.device)

        train_data = EasyData(root_dir="data/train", transform=self.train_transform, grayscale=self.grayscale)
        test_data = EasyData(root_dir="data/test", transform=self.test_transform, grayscale=self.grayscale)

        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

        for epoch in range(total_epochs):
            print("** EPOCH", (epoch+1), "**")
            self.__train_one_epoch(train_loader, loss_fn, optimizer)
            print("TRAINING:")
            self.__test(train_loader, loss_fn) 
            print("TESTING:")
            self.__test(test_loader, loss_fn)  

        if save_model:
            torch.save(self.model.state_dict(), model_name + ".pth")
            print("model saved as " + model_name +".pth")


class EasyData:
    def __init__(self, root_dir, transform=None, grayscale=False):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        self.grayscale = grayscale

        self.images = []
        self.labels = []

        for i, class_name in enumerate(self.classes):
            class_dir = os.path.join(root_dir, class_name)
            for img_name in os.listdir(class_dir):
                self.images.append(os.path.join(class_dir, img_name))
                self.labels.append(i)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        label = self.labels[idx]
        if not self.grayscale:
            image = Image.open(img_name)
        else: 
            image = Image.open(img_name).convert('L')
        if self.transform:
            image = self.transform(image)

        return image, label

class ExampleModelV1(nn.Module):
    def __init__(self, class_cnt, channel_cnt, img_size):
        super().__init__()
        self.net_stack = nn.Sequential(
            nn.Conv2d(channel_cnt, img_size, 3, padding="same"),
            nn.BatchNorm2d(img_size),
            nn.ReLU(),
            nn.Conv2d(img_size ,64, 3, padding="same"),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, 3),
            
            nn.Conv2d(64, 128, 3, padding="same"),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding="same"),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 128, 3, padding="same"),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding="same"),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 128, 3, padding="same"),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, padding="same"),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(256, 128, 3, padding="same"),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding="same"),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 32, 3, padding="same"),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Flatten(),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            
            nn.Linear(32, class_cnt) # Output layer
        )

    def forward(self, x):
        logits = self.net_stack(x)
        return logits
    
class ExampleModelV2(nn.Module):
    def __init__(self, class_cnt, channel_cnt, img_size):
        super().__init__()
        self.net_stack = nn.Sequential(
            nn.Conv2d(channel_cnt, img_size, 3, padding="same"),
            nn.BatchNorm2d(img_size),
            nn.ReLU(),
            nn.Conv2d(img_size ,64, 3, padding="same"),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, 3, padding="same"),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding="same"),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 128, 3, padding="same"),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding="same"),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Flatten(),

            nn.Linear(16384, 32),
            nn.BatchNorm1d(32),
            nn.Linear(32, 32),
            nn.BatchNorm1d(32),
            
            nn.Linear(32, class_cnt) # Output layer
        )

    def forward(self, x):
        logits = self.net_stack(x)
        return logits
    
# https://blog.paperspace.com/vgg-from-scratch-pytorch/
class VGG16(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG16, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.layer6 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.layer7 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer8 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer9 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer10 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer11 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer12 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer13 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(7*7*512, 4096),
            nn.ReLU())
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU())
        self.fc2= nn.Sequential(
            nn.Linear(4096, num_classes))
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        out = self.layer9(out)
        out = self.layer10(out)
        out = self.layer11(out)
        out = self.layer12(out)
        out = self.layer13(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out
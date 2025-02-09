import time
import torch
import torch.nn as nn
from torchvision import datasets, transforms


class Model(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels = 1,
            out_channels = 8,
            kernel_size = 3,
            stride = 1,
            padding = 1,
        )
        self.conv2 = nn.Conv2d(
            in_channels=8,
            out_channels = 16,
            kernel_size = 3,
            stride = 1,
            padding = 1
        )
        self.conv3 = nn.Conv2d(
            in_channels = 16,
            out_channels = 32,
            kernel_size = 3,
            stride = 1,
            padding = 1
        )

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.maxpool = nn.MaxPool2d(
            kernel_size = 2,
            stride  = 2,
        )
        self.linear1 = nn.Linear(32*14*14, 10)
        #self.linear2 = nn.Linear(2056,10)
        self.sigmoid = nn.Sigmoid()


    def forward(self,x):

        x = self.conv1(x)
        x =self.relu(x)
        # x = self.dropout(x) # makes performance absolutely worse
        x = self.relu(self.conv2(x))
        x = self.dropout(x)
        x = self.relu(self.conv3(x))
        #x = self.dropout(x)
        x = self.maxpool(x)
        
        x = x.flatten(1)
        #x = self.dropout(x)  this does not work - acc - 9%
        # decreasing the nn model filter structure from 32-64-128 to 8-16-32 improved performance to a great degree

        x = self.linear1(x)
        # x = self.relu(self.linear1(x))
        # x = self.dropout(x)
        # x = self.linear2(x)
        x = self.sigmoid(x)

        return x


def train(epochs):

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST('data',train=True,download=True,transform=transform)
    test_dataset = datasets.MNIST('data',train=False,download=True,transform=transform)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size = 32,
        shuffle = True
    ) 

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=32
    )

    model = Model()
    loss_fx = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr = 1e-3)

    start_time = time.time()
    for epoch in range(epochs):
        model.train()
        loss_val = 0
        correct = 0
        total = 0

        for batch_idx,(data,target) in enumerate(train_dataloader):
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fx(output,target)
            loss.backward()
            optimizer.step()

            loss_val += loss.item()
            _, predicted = torch.max(output.data, 1)  # Get the predicted class
            total += target.size(0)  # Add batch size
            correct += (predicted == target).sum().item() 
            #print(f'Epoch {epoch}:  batch {batch_idx}: loss - {loss.item():.4f}')

        epoch_loss_avg = loss_val/ len(train_dataloader)
        accuracy = 100 * correct / total
        
        print(f'Epoch {epoch}/{epochs-1}, Loss: {epoch_loss_avg:.4f},  Accuracy: {accuracy:.2f}%')
    end_time = time.time()
    print(f'Train time: {end_time - start_time:.2f} seconds')


        


if __name__ == "__main__" :

    start = time.time()
    train(5)
    end = time.time()
    print(f'Runtime:')
    print(f'Total runtime: {end - start:.2f} seconds')




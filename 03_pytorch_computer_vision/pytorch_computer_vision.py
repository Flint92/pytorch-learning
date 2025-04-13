import torch
import time
import matplotlib.pyplot as plt

from torch import nn
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from utils.help_functions import set_seeds, accuracy_fn
from utils.device import get_device
from tqdm.auto import tqdm


class FashionMNISTModelV0(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=input_shape,
                      out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units,
                      out_features=output_shape),
            # nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer_stack(x)


BATCH_SIZE = 32


if __name__ == '__main__':
    train_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
        target_transform=None,
    )

    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
        target_transform=None,
    )

    class_names = train_data.classes
    print(class_names)

    class_to_idx = train_data.class_to_idx
    print(class_to_idx)

    # image, label = train_data[0]
    # print(f'Image shape: {image.shape} -> [color_channels, height, width]')
    # print(f'Image label: {class_names[label]}')
    #
    # plt.imshow(image.squeeze(), cmap=plt.colormaps.get_cmap('gray'))
    # plt.title(class_names[label])
    # plt.axis(False)
    # plt.show()

    set_seeds()
    plt.figure(figsize=(9, 9))
    rows, cols = 4, 4
    for i in range(1, rows * cols + 1):
        random_idx = torch.randint(0, len(train_data), size=[1]).item()
        item = train_data[random_idx]
        img, label = item
        plt.subplot(rows, cols, i)
        plt.imshow(img.squeeze(), cmap=plt.colormaps.get_cmap('gray'))
        plt.title(class_names[label])
        plt.axis(False)
    plt.show()

    # Turn datasets into iterables(batched)
    train_data_loader = DataLoader(dataset=train_data,
                                   batch_size=BATCH_SIZE,
                                   shuffle=True,
                                   )

    test_data_loader = DataLoader(dataset=test_data,
                                  batch_size=BATCH_SIZE,
                                  shuffle=False,
                                  )

    train_features_batch, train_labels_batch = next(iter(train_data_loader))

    flatten_model = nn.Flatten()
    x = train_features_batch[0]
    output = flatten_model(x)
    print(f'Shape before flattening: {x.shape} -> [color_channels, height, width]')
    print(f'Shape after flattening: {output.shape} -> [color_channels, height * width]')

    device = get_device()

    set_seeds()
    model = FashionMNISTModelV0(input_shape=28 * 28,
                                hidden_units=10,
                                output_shape=len(class_names)
                                ).to(device)

    loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(params=model.parameters(),
                                lr=0.1)

    set_seeds()

    epochs = 3
    start_time = time.time()
    for epoch in tqdm(range(epochs)):
        print(f'Epoch: {epoch}\n-------')

        train_loss, train_acc = 0, 0
        for batch, (X, y) in enumerate(train_data_loader):
            model.train()

            X, y = X.to(device), y.to(device)

            y_pred = model(X)

            loss = loss_fn(y_pred, y)

            train_loss += loss
            train_acc += accuracy_fn(y_true=y, y_pred=y_pred.argmax(dim=1))

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            if batch % 400 == 0:
                print(f'Looked at {batch * len(X)}/{len(train_data)} samples')

        train_loss /= len(train_data_loader)
        train_acc /= len(train_data_loader)
        print(f'\nTrain loss: {train_loss:.5f} | Train accuracy: {train_acc:.2f}%\n')

    end_time = time.time()
    print(f'Total training time on device: {device} is {end_time - start_time:.3f} seconds')

    test_loss, test_acc = 0, 0
    model.eval()
    with torch.inference_mode():
        for X_test, y_test in test_data_loader:
            X_test, y_test = X_test.to(device), y_test.to(device)
            test_pred = model(X_test)
            test_loss += loss_fn(test_pred, y_test)
            test_acc += accuracy_fn(y_true=y_test, y_pred=test_pred.argmax(dim=1))
        test_loss /= len(test_data_loader)
        test_acc /= len(test_data_loader)
        print(f'\nTest loss: {test_loss:.5f} | Test accuracy: {test_acc:.2f}%\n')
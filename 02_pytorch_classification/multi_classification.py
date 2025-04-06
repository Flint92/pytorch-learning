import torch
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from torch import nn # nn contains all of PyTorch's building blocks for neural networks
from torchmetrics import Accuracy
from utils.help_functions import accuracy_fn, plot_decision_boundary
from utils.device import get_device


class BlobModel(nn.Module):
    def __init__(self, input_features, output_features, hidden_units=8):
        super().__init__()
        self.linear_layer_stack = nn.Sequential(
            nn.Linear(in_features=input_features, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=output_features)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_layer_stack(x)



NUM_CLASSES = 4
NUM_FEATURES = 2
RANDOM_SEED = 42

if __name__ == '__main__':
    device = get_device()

    # 1. Create multi-class data
    X_blob, y_blob = make_blobs(n_samples=1000,
                                n_features=NUM_FEATURES,
                                centers=NUM_CLASSES,
                                cluster_std=1.5,
                                random_state=RANDOM_SEED)

    # 2. Turn data into tensors
    X_blob = torch.from_numpy(X_blob).type(torch.float)
    y_blob = torch.from_numpy(y_blob).type(torch.long)

    # 3. Split into train and test sets
    X_blob_train, X_blob_test, y_blob_train, y_blob_test = train_test_split(X_blob,
                                                                            y_blob,
                                                                            test_size=0.2,
                                                                            random_state=RANDOM_SEED)
    X_blob_train,  y_blob_train = X_blob_train.to(device), y_blob_train.to(device)
    X_blob_test, y_blob_test = X_blob_test.to(device), y_blob_test.to(device)

    # 4. Plot
    plt.figure(figsize=(10, 7))
    plt.scatter(X_blob[:, 0], X_blob[:, 1], c=y_blob, cmap=plt.get_cmap('RdYlBu'))
    plt.show()

    model = BlobModel(input_features=2,
                      output_features=4,
                      hidden_units=8).to(device)


    # 5. Create loss function
    loss_fn = nn.CrossEntropyLoss()

    # 6. Create Optimizer
    optimizer = torch.optim.Adam(params=model.parameters(),
                                lr=0.1)

    torchmetrics_accuracy = Accuracy(task='multiclass', num_classes=NUM_CLASSES).to(device)

    # model.eval()
    # with torch.inference_mode():
    #     y_test_logits = model(X_blob_test)
    #     y_test_pred_probs = torch.softmax(y_test_logits, dim=1)
    # print((y_test_logits[:5], y_test_pred_probs[:5]))

    epochs = 200
    for epoch in range(epochs):
        model.train()

        y_logits = model(X_blob_train)
        y_preds = torch.softmax(y_logits, dim=1).argmax(dim=1)

        loss = loss_fn(y_logits, y_blob_train)
        # acc = accuracy_fn(y_true=y_blob_train,
        #                   y_pred=y_preds)
        acc = torchmetrics_accuracy(y_preds, y_blob_train)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        model.eval()
        with torch.inference_mode():
            y_test_logits = model(X_blob_test)
            y_test_preds = torch.softmax(y_test_logits, dim=1).argmax(dim=1)
            test_loss = loss_fn(y_test_logits, y_blob_test)
            # test_acc = accuracy_fn(y_true=y_blob_test,
            #                        y_pred=y_test_preds)
            test_acc = torchmetrics_accuracy(y_test_preds, y_blob_test)

        if epoch % 10 == 0:
            print(f'Epoch: {epoch} | Loss: {loss:.5f}, Acc: {acc:.2f} | Test Loss: {test_loss:.5f}, Test Acc: {test_acc:.2f}')


    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title('Train')
    plot_decision_boundary(model, X_blob_train, y_blob_train)
    plt.subplot(1, 2, 2)
    plt.title('Test')
    plot_decision_boundary(model, X_blob_test, y_blob_test)
    plt.show()

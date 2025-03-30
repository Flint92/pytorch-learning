import pandas as pd
import matplotlib.pyplot as plt
import torch

from sklearn import datasets
from sklearn.model_selection import train_test_split
from torch import nn
from utils.device import get_device
from utils.help_functions import plot_predictions, plot_decision_boundary

class CircleModelV0(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=2, out_features=5)
        self.layer_2 = nn.Linear(in_features=5, out_features=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer_2(self.layer_1(x))


def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc

if __name__ == '__main__':
    # Make 1000 samples
    n_samples = 1000
    # Create Circles
    X, y = datasets.make_circles(n_samples, noise=0.03, random_state=42)

    # print(len(X), len(y))
    # print(f'First 5 samples of X:\n{X[:5]}')
    # print(f'First 5 samples of y:\n{y[:5]}')

    # Make dataframe of circle data
    circles = pd.DataFrame({
        'X1': X[:, 0],
        'X2': X[:, 1],
        'label': y
    })
    print(circles.head(5))

    # Visualize with a plot
    plt.scatter(x=X[:, 0],
                y=X[:, 1],
                c=y,
                cmap=plt.colormaps.get_cmap('RdYlBu'))
    plt.show()

    # Transform data into tensors and create train and test splits
    X = torch.from_numpy(X).type(torch.float)
    y = torch.from_numpy(y).type(torch.float)

    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.2,
                                                        random_state=42)


    device = get_device()
    # model_0 = CircleModelV0().to(device)
    # print(model_0.state_dict())


    model_0 = nn.Sequential(
        nn.Linear(in_features=2, out_features=5),
        nn.Linear(in_features=5, out_features=1),
    ).to(device)
    # print(model_0.state_dict())

    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(params=model_0.parameters(),
                                lr=0.1)

    # with torch.inference_mode():
    #     model_0.eval()
    #     y_logits = model_0(X_test.to(device))[:5]
    #     y_pred_probs = torch.sigmoid(y_logits)
    #     y_preds = torch.round(y_pred_probs)
    #     y_pred_labels = torch.round(torch.sigmoid(model_0(X_test.to(device))[:5]))
    #     print(torch.eq(y_preds.squeeze(), y_pred_labels.squeeze()))

    torch.manual_seed(42)  # set the seed for CPU and GPU
    torch.mps.manual_seed(42)  # set the seed for MPS
    torch.cuda.manual_seed(42)  # set the seed for CUDA

    epoches = 180
    for epoch in range(epoches):
        model_0.train()

        y_logits = model_0(X_train.to(device)).squeeze()
        y_preds = torch.round(torch.sigmoid(y_logits))

        loss = loss_fn(y_logits, y_train.to(device))
        acc = accuracy_fn(y_train.to(device), y_preds)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        model_0.eval()
        with torch.inference_mode():
            test_logits = model_0(X_test.to(device)).squeeze()
            test_preds = torch.round(torch.sigmoid(test_logits))
            test_loss = loss_fn(test_logits, y_test.to(device))
            test_acc = accuracy_fn(y_test.to(device), test_preds)

        if epoch % 10 == 0:
            print(f'Epoch: {epoch} | Loss: {loss:.5f}, Accuracy: {acc:.2f}% | Test Loss: {test_loss:.5f}, Test Accuracy: {test_acc:.2f}%')


    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title('Train')
    plot_decision_boundary(model_0, X_train, y_train)
    plt.subplot(1, 2, 2)
    plt.title('Test')
    plot_decision_boundary(model_0, X_test, y_test)
    plt.show()


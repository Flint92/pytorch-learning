import torch
import matplotlib.pyplot as plt

from torch import nn # nn contains all of PyTorch's building blocks for neural networks
from pathlib import Path



def plot_predictions(train_data, train_labels, test_data, test_labels, predictions=None):
    """
    Plots training data, testing data and compares predictions
    """

    plt.figure(figsize=(10, 7))

    # plot training data in blue
    plt.scatter(train_data, train_labels, c='b', s=4, label="Training data")

    # plot testing data in green
    plt.scatter(test_data, test_labels, c='g', s=4, label="Testing data")

    if predictions is not None:
        # plot the predictions in red
        plt.scatter(test_data, predictions, c='r', s=4, label="Predictions")

    # show the legend
    plt.legend(prop={'size': 14})
    plt.show()

# create linear regression model class
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(1,
                                            requires_grad=True,
                                            dtype=torch.float))
        self.bias = nn.Parameter(torch.randn(1,
                                             requires_grad=True,
                                             dtype=torch.float))
    # define the forward computation (input data x -> output data)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weights * x + self.bias


if __name__ == '__main__':
    # create *known* parameters
    weight = 0.7
    bias = 0.3

    # create data
    start = 0
    end = 1
    step = 0.02
    X = torch.arange(start, end, step).unsqueeze(dim=1)
    y = weight * X + bias

    # create train/test split
    train_split = int(0.8 * len(X))
    X_train, y_train = X[:train_split], y[:train_split]
    X_test, y_test = X[train_split:], y[train_split:]

    # create a random seed
    torch.manual_seed(42)
    # create an instance of the model
    model_0 = LinearRegressionModel()
    # check out
    # print(list(model_0.parameters()))
    # listed named parameters
    print(model_0.state_dict())

    # make predictions with model
    with torch.inference_mode(): # inference mode is a context manager, it will turn off gradient tracking
        y_preds = model_0(X_test)

    # # you can also do something similar with torch.no_grad(), however torch.inference_mode() is preferred
    # with torch.no_grad():
    #     y_preds = model_0(X_test)

    plot_predictions(train_data=X_train.numpy(), train_labels=y_train.numpy(),
                     test_data=X_test.numpy(), test_labels=y_test.numpy(), predictions=y_preds.numpy())

    # setup a loss function
    loss_fn = nn.L1Loss() # MAE loss is also called L1 loss
    # setup an optimizer
    optimizer = torch.optim.SGD(params=model_0.parameters(),
                                lr=0.01) # lr = learning rate = possible most important hyperparameter you can set
    # build a training loop
    epochs = 200
    epoch_count = []
    train_loss_values = []
    test_loss_values = []
    # 0. loop through the data
    for epoch in range(epochs):
        # set the model to training model
        model_0.train()

        # 1. Forward pass
        y_pred = model_0(X_train)

        # 2. Calculate loss
        loss = loss_fn(y_pred, y_train)


        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Perform backpropagation on the loss with respect to the parameters of the model
        loss.backward()

        # 5. step the optimizer (perform gradient descent)
        optimizer.step() # by default how the optimizer changes will accumulate through the loop so we have to zero them above in step 3 for the next iteration of the loop

        # Testing
        model_0.eval() # turn off different settings in the model not needed for evaluation/testing (dropout/batch norm)
        with torch.inference_mode(): # turn off gradient tracking and a couple more things behind the scenes
            # 1. Forward pass
            test_pred = model_0(X_test)
            # 2. Calculate loss
            test_loss = loss_fn(test_pred, y_test)
            if epoch % 10 == 0:
                epoch_count.append(epoch)
                train_loss_values.append(loss.detach().numpy())
                test_loss_values.append(test_loss.detach().numpy())

    with torch.inference_mode():
        y_preds_new = model_0(X_test)
    plot_predictions(train_data=X_train.numpy(), train_labels=y_train.numpy(),
                     test_data=X_test.numpy(), test_labels=y_test.numpy(), predictions=y_preds_new.numpy())

    plt.plot(epoch_count, train_loss_values, label="Train loss")
    plt.plot(epoch_count, test_loss_values, label="Test loss")
    plt.title("Training and test loss curves")
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.legend()
    plt.show()


    ## Saving a model in PyTorch
    # 1. Create models directory
    MODEL_PATH = Path("models")
    MODEL_PATH.mkdir(parents=True, exist_ok=True)

    # 2. Create model save path
    MODEL_NAME = "01_pytorch_workflow_model_0.pth"
    MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

    # 3. Save the model state dict
    print(f"Saving model to: {MODEL_SAVE_PATH}")
    torch.save(obj=model_0.state_dict(), # only saving the state_dict() only saves the models learned parameters
               f=MODEL_SAVE_PATH)

    ## Loading a PyTorch model
    loaded_model_0 = LinearRegressionModel()
    loaded_model_0.load_state_dict(torch.load(f=MODEL_SAVE_PATH, weights_only=False))
    print(loaded_model_0.state_dict())

    loaded_model_0.eval()
    with torch.inference_mode():
        loaded_model_preds = loaded_model_0(X_test)

    print(loaded_model_preds == y_preds_new)


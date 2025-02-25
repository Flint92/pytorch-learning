import torch

if __name__ == '__main__':
    # scalar
    scalar = torch.tensor(7)
    print((scalar, scalar.ndim, scalar.shape, scalar.item()))

    # vector
    vector = torch.tensor([7, 7])
    print((vector, vector.ndim, vector.shape))

    # matrix
    MATRIX = torch.tensor([[7, 8],
                           [9, 10]])
    print((MATRIX, MATRIX.ndim, MATRIX.shape))
    print((MATRIX[0], MATRIX[1]))

    # tensor
    TENSOR = torch.tensor([[[1, 2, 3],
                            [4, 5, 6],
                            [7, 8, 9]],

                           [[10, 11, 12],
                           [13, 14, 15],
                           [16, 17, 18]]])
    print((TENSOR, TENSOR.ndim, TENSOR.shape))

    # random tensor
    random_tensor = torch.randn(size=(2, 3, 4))
    print(random_tensor)

    random_image_size_tensor = torch.randn(size=(3, 224, 224))
    print(random_image_size_tensor.shape, random_image_size_tensor.ndim)

    # zeros and ones
    zeros = torch.zeros(size=(3, 4))
    ones = torch.ones(size=(3, 4))
    print((zeros, ones))

    # range tensor and  tensor like
    range_tensor = torch.arange(start=0, end=10, step=1)
    ten_zeros = torch.zeros_like(input=range_tensor)
    print(range_tensor, ten_zeros)

    # matrix multiplication
    tensor_A = torch.tensor([[1, 2],
                             [3, 4],
                             [5, 6]])
    tensor_B = torch.tensor([[7, 8, 9],
                             [10, 11, 12]])
    print(tensor_A @ tensor_B) # torch.matmul(tensor_A, tensor_B)

    # min, max, mean, sum
    x = torch.arange(0, 100, 10)
    print((x.min(), x.max(), x.type(torch.float32).mean(), x.sum()))

    # position of min and max
    print((x.argmin(), x.argmax()))

    # reshaping, stacking, squeezing, unsqueezing tensors
    x = torch.arange(1., 10.)
    print((x, x.shape))

    x_reshaped = x.reshape((3, 3)) # add an extra dimension
    print(x_reshaped)

    z = x.view(1, 9)
    print((z, z.shape))
    z[:, 0] = 5
    print((z, x, x_reshaped)) # view and reshape share the same memory space

    # stack tensors on top of each other
    x_stacked = torch.stack([x_reshaped, x_reshaped, x_reshaped, x_reshaped], dim=1)
    print((x_stacked, x_stacked.shape))

    # squeeze -> remove all single dims or a single dim at specific dim from a target tensor
    x = torch.randn(1, 3, 1, 4)
    x_squeezed = x.squeeze(dim=2)
    print((x.shape, x_squeezed.shape))  # (torch.Size([1, 3, 1, 4]), torch.Size([1, 3, 4]))

    # unsqueeze -> adds a single dim to a target tensor at a specific dim
    x_unsqueezed = x_squeezed.unsqueeze(dim=1)
    print((x_squeezed.shape, x_unsqueezed.shape))

    # permute -> rearrange the dimensions of a target tensor in a specific order
    x = torch.arange(1, 10)
    x_reshaped = x.reshape(1, 3, 3)
    print((x_reshaped, x_reshaped.permute(1, 0, 2)))

    # random seed
    RANDOM_SEED = 42

    torch.manual_seed(RANDOM_SEED)
    random_tensor_C = torch.randn((3, 4))

    torch.manual_seed(RANDOM_SEED)
    random_tensor_D = torch.randn((3, 4))

    print(random_tensor_C == random_tensor_D)
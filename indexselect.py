import torch

def indices_select(input:torch.Tensor, labels:torch.Tensor, index:torch.Tensor):
    r""" Mustafa Mohammadi
    This function return the samples with specific labels:
    :param input: torch.Tensor
    :param labels: torch.Tensor
    :param index: optional[int, torch.Tensor]
    :return: samples: 2D torch.Tensor, 1D torch.Tensor indices

    example:
    samples = torch.randn((20, 5), dtype=torch.float64)
    labels = torch.randint(0, 9, (20, 1))

    output, indices = indices_select(input=samples, labels=labels, index=4)
    print(output)
    print(indices)
    """

    index = torch.tensor(index)
    indices = labels == index
    indices = indices.nonzero()
    return input[indices[:, 0], :], indices[:, 0]

samples = torch.randn((20, 5), dtype=torch.float64)
labels = torch.randint(0, 9, (20, 1))

print(indices_select(samples, labels, 5))




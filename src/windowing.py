import torch


def window_partition_2d(x: torch.Tensor, window_size: int):
    N, C, H, W = x.shape
    x = x.view(
        N,
        C, 
        H // window_size, window_size,
        W // window_size, window_size
    )
    return x.permute(
        0, 1, 2, 4, 3, 5
    ).contiguous().view(
        N, -1, window_size, window_size
    )


def window_reverse_2d(windows: torch.Tensor, window_size: int, H: int, W: int):
    N = windows.shape[0]
    C = int(
        windows.shape[1] / (H * W / window_size / window_size)
    )
    x = windows.view(
        N, 
        C,
        H // window_size,
        W // window_size,
        window_size, 
        window_size
    )
    return x.permute(
        0, 1, 2, 4, 3, 5
    ).contiguous().view(
        N, C, H, W
    )


def window_partition_3d(x: torch.Tensor, window_size: int):
    N, C, H, W, D = x.shape
    x = x.view(
        N, 
        C, 
        H // window_size, window_size, 
        W // window_size, window_size, 
        D // window_size, window_size
    )
    return x.permute(
        0, 1, 2, 4, 6, 3, 5, 7
    ).contiguous().view(
        N, -1, window_size, window_size, window_size
    )


def window_reverse_3d(
        windows: torch.Tensor,
        window_size: int,
        H: int,
        W: int,
        D: int
    ):
    N = windows.shape[0]
    C = int(
        windows.shape[1] / (H * W * D / window_size / window_size / window_size)
    )
    x = windows.view(
        N, 
        C,
        H // window_size, 
        W // window_size, 
        D // window_size, 
        window_size, 
        window_size, 
        window_size
    )
    return x.permute(
        0, 1, 2, 5, 3, 6, 4, 7
    ).contiguous().view(
        N, C, H, W, D
    )

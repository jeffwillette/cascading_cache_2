import torch


def window_split(x, window_size):
    """
    input: tensor of size (B, S, D)
    output: tensor of size (B, S - W + 1, W, D)

    creates a window seqeuence which is stored in the 3rd dimension
    """
    B, S, D = x.size()
    # print(f"{x.size()=}")

    if window_size > S - 1:
        raise ValueError("window size must be smaller than S - 1 to be valid")

    s_idx = torch.arange(window_size).view(1, window_size).repeat(
        S - window_size + 1, 1)
    s_idx += torch.arange(S - window_size + 1).unsqueeze(-1)

    # for S=10 and window_size=5, it looks like this
    # print(s_idx)
    # tensor([[0, 1, 2, 3, 4],                                                                                                                                                          │···························································
    #         [1, 2, 3, 4, 5],                                                                                                                                                          │···························································
    #         [2, 3, 4, 5, 6],                                                                                                                                                          │···························································
    #         [3, 4, 5, 6, 7],                                                                                                                                                          │···························································
    #         [4, 5, 6, 7, 8],                                                                                                                                                          │···························································
    #         [5, 6, 7, 8, 9]])

    s_idx = s_idx.view(1, S - window_size + 1,
                       window_size).repeat(B, 1, 1).view(B, -1)
    b_idx = torch.arange(B).view(B, 1).repeat(1, s_idx.size(1))

    x = x[b_idx, s_idx]
    # print(x)
    x = x.reshape(B, S - window_size + 1, window_size, D)
    # print(x.shape)

    return x


if __name__ == "__main__":
    S = 10
    w = 5
    out = window_split(torch.arange(S).view(1, S, 1).repeat(2, 1, 3), w)
    print(out.size())

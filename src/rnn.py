import torch
import torch.nn as nn
    

def kwta(h, k):
    values, indices = torch.topk(h, k, dim=1)
    mask = torch.zeros_like(h)
    mask.scatter_(1, indices, 1.0)
    return h * mask


class RNN(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, hidden_loops=0, k=None,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.hidden_loops = hidden_loops
        self.k = k

        self.in_lin = nn.Linear(input_dim, hidden_dim, bias=True)
        self.hid_lin = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out_lin = nn.Linear(hidden_dim, output_dim, bias=True)

        if k:
            self.act = lambda x: kwta(nn.functional.relu(x), k)
        else:
            self.act = lambda x: nn.functional.relu(x)

    def forward(self, x, h=None, return_all_h=False):
        B, T, F = x.shape

        if h is None:
            h = torch.zeros(B, self.hidden_dim, device=x.device, dtype=x.dtype)

        hs = []
        # Pre-allocate h_all to avoid the memory spike of torch.stack
        h_all = None
        if return_all_h:
            h_all = torch.empty(B, T, 1 + self.hidden_loops, self.hidden_dim, device=x.device, dtype=x.dtype)

        for t in range(T):
            x_t = x[:, t, :]
            
            # Initial transition from input + previous state
            h = self.act(self.in_lin(x_t) + self.hid_lin(h))
            if return_all_h:
                h_all[:, t, 0] = h

            # Recurrent hidden loops (updates h in-place within the timestep)
            for i in range(self.hidden_loops):
                h = self.act(self.hid_lin(h))
                if return_all_h:
                    h_all[:, t, i+1] = h

            hs.append(h)

        h_seq = torch.stack(hs, dim=1)          # (B, T, hidden_dim)
        y = self.out_lin(h_seq)

        return y, h_seq, h_all


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    m = RNN(4, 64, 2, k=8)
    x = torch.randn(32, 16, 4)
    y, hs = m(x)
    plt.imshow(hs[0].detach(), vmin=0)
    plt.show()

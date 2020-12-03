import torch
from .common import MLP, SumRateLoss
import time


class Net(torch.nn.Module):
    def __init__(self,
                 n_inputs,
                 n_outputs,
                 n_tasks,
                 args):
        super(Net, self).__init__()

        # setup network
        hidden = [int(x) for x in args.hidden_layers.split("-")]
        self.net = MLP([n_inputs] + hidden + [n_outputs])
        self.nc_per_task = n_outputs
        self.n_outputs = n_outputs

        # setup optimizer
        self.opt = torch.optim.RMSprop(self.parameters(), lr=args.lr)
        self.n_iter = args.n_iter
        self.mini_batch_size = args.mini_batch_size

        # setup losses
        self.noise = args.noise
        self.loss = torch.nn.MSELoss()
        self.loss_sumrate = SumRateLoss

    def forward(self, x, t):
        output = self.net(x)
        return output

    def observe(self, x, t, y, loss_type='MSE', x_te=None, x_tr=None):
        self.train()
        time_spent = 0
        for epoch in range(self.n_iter):
            permutation = torch.randperm(x.size()[0])
            for i in range(0, x.size()[0], self.mini_batch_size):
                time_start = time.time()
                self.zero_grad()
                indices = permutation[i:i + self.mini_batch_size]
                batch_x, batch_y = x[indices], y[indices]
                if loss_type == 'MSE':
                    ptloss = self.loss(self.forward(batch_x, t), batch_y)
                else:
                    ptloss = self.loss_sumrate(
                        batch_x, self.forward(batch_x, t), self.noise)
                ptloss.backward()
                self.opt.step()
                time_end = time.time()
                time_spent = time_spent + time_end - time_start

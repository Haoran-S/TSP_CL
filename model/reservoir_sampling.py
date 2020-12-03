import torch
from .common import MLP, SumRateLoss
import random


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

        # allocate buffer
        self.M = []
        self.age = 0
        self.memories = args.n_memories

    def forward(self, x, t):
        output = self.net(x)
        return output

    def get_batch(self, x, y):
        if self.M:
            # combine buffer with current samples
            set_x = torch.stack([self.M[k][0] for k in range(len(self.M))], 0)
            set_y = torch.stack([self.M[k][1] for k in range(len(self.M))], 0)
            set_x = torch.cat([set_x, x], 0)
            set_y = torch.cat([set_y, y], 0)
            return set_x, set_y
        else:
            return x, y

    def observe(self, x, t, y, loss_type='MSE', x_tr=None, x_te=None):
        self.train()
        set_x, set_y = self.get_batch(x, y)
        for epoch in range(self.n_iter):
            permutation = torch.randperm(set_x.size()[0])
            for i in range(0, x.size()[0], self.mini_batch_size):
                self.zero_grad()
                indices = permutation[i:i + self.mini_batch_size]
                batch_x, batch_y = set_x[indices], set_y[indices]
                if loss_type == 'MSE':
                    ptloss = self.loss(self.forward(batch_x, t), batch_y)
                else:
                    ptloss = self.loss_sumrate(
                        batch_x, self.forward(batch_x, t), self.noise)
                ptloss.backward()
                self.opt.step()

        for i in range(0, x.size()[0]):
            self.age += 1
            if len(self.M) < self.memories:
                # add new samples to the buffer
                self.M.append([x[i], y[i], t])
            else:
                # buffer is full
                p = random.randint(0, self.age)
                if p < self.memories:
                    self.M[p] = [x[i], y[i], t]

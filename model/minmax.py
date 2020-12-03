import torch
from .common import MLP, weighted_mse_loss, weighted_sumrate_loss
from .common import weighted_ratio_loss, mse_per_sample
import numpy as np


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
        self.loss_wmse = weighted_mse_loss
        self.loss_wsumrate = weighted_sumrate_loss
        if args.eval_metric == 'mse':
            self.loss_dual = mse_per_sample
        elif args.eval_metric == 'ratio':
            self.loss_dual = weighted_ratio_loss
        else:
            raise AssertionError('error')

        # allocate buffer
        self.M = []
        self.age = 0
        self.memories = args.n_memories
        self.dual_stepsize = args.dual_stepsize
        self.weight_ini = args.weight_ini

    def forward(self, x, t):
        output = self.net(x)
        return output

    def get_batch(self, x, y):
        if self.M:
            # combine buffer with current samples
            set_x, set_y = self.M
            set_x = torch.cat([set_x, x], 0)
            set_y = torch.cat([set_y, y], 0)
        else:
            set_x, set_y = x, y
        batch_size = set_x.size()[0]

        if self.weight_ini == 'pra':
            set_w = torch.ones(batch_size)
            set_w[0:self.memories] = self.age / self.memories
            self.age += x.size()[0]
        elif self.weight_ini == 'mean':
            set_w = torch.ones(batch_size) / batch_size
        else:
            set_w = torch.rand(batch_size)
        return set_x, set_y, set_w

    def MSE_per_sample(self, input, target):
        return torch.sum((input - target) ** 2, 1)

    def proj(self, v, z=1):
        v = v.detach().numpy()
        n_features = v.shape[0]
        u = np.sort(v)[::-1]
        cssv = np.cumsum(u) - z
        ind = np.arange(n_features) + 1
        cond = u - cssv / ind > 0
        rho = ind[cond][-1]
        theta = cssv[cond][-1] / float(rho)
        w = np.maximum(v - theta, 0)
        return torch.from_numpy(w)

    def observe(self, x, t, y, loss_type='MSE', x_te=None, x_tr=None):
        self.train()
        set_x, set_y, set_w = self.get_batch(x, y)

        for epoch in range(self.n_iter):
            permutation = torch.randperm(set_x.size()[0])
            for i in range(0, x.size()[0], self.mini_batch_size):
                # primal
                self.zero_grad()
                indices = permutation[i:i + self.mini_batch_size]
                batch_x = set_x[indices]
                batch_y = set_y[indices]
                batch_w = set_w[indices]

                if loss_type == 'MSE':
                    ptloss = self.loss_wmse(
                        self.forward(batch_x, t), batch_y, batch_w)
                elif loss_type == 'SUMRATE':
                    ptloss = self.loss_wsumrate(
                        batch_x, self.forward(batch_x, t), batch_w, self.noise)
                ptloss.backward()
                self.opt.step()

            # dual
            set_w = set_w + self.dual_stepsize * \
                self.loss_dual(set_x, self.forward(
                    set_x, t), set_y, self.noise)
            set_w = self.proj(set_w)

        _, indices = torch.sort(set_w, descending=True)
        if len(set_w) <= self.memories:
            self.M = (set_x, set_y)
        else:
            self.M = (set_x[indices[0:self.memories]],
                      set_y[indices[0:self.memories]])

        print(torch.sum(indices[0:self.memories] < self.memories).item(
        ), ' out of ', self.memories, ' samples in buffer are keeped')

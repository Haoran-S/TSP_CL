import torch
from .common import MLP, weighted_mse_loss, weighted_sumrate_loss
from .common import weighted_ratio_loss, mse_per_sample, SumRateLoss


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
        self.lr = args.lr
        self.n_iter = args.n_iter
        self.mini_batch_size = args.mini_batch_size

        # setup losses
        self.noise = args.noise
        self.loss_wmse = weighted_mse_loss
        self.loss_dual = weighted_ratio_loss

        # allocate buffer
        self.M = []
        self.age = 0
        self.memories = args.n_memories
        
        self.set_index = []

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
        return set_x, set_y

    def MSE_per_sample(self, input, target):
        return torch.sum((input - target) ** 2, 1)

    def observe(self, x, t, y, loss_type='MSE', x_te=None, x_tr=None, scale = 1.0):
        self.train()
        set_x, set_y = self.get_batch(x, y)
        grads_squared = [0.0 for _ in self.net.parameters()]
        self.age += 1
        for epoch in range(self.n_iter):                
            permutation1 = torch.randperm(set_x.size()[0])
            permutation2 = torch.randperm(set_x.size()[0])
            weight_scale = set_x.size()[0] / self.mini_batch_size
            for i in range(0, x.size()[0], self.mini_batch_size):
                # primal
                self.zero_grad()
                indices1 = permutation1[i:i + self.mini_batch_size]
                batch_x1 = set_x[indices1]
                batch_y1 = set_y[indices1]
                                
                g_loss = torch.sum(torch.exp(scale*self.loss_dual(batch_x1, self.forward(batch_x1, t), batch_y1, self.noise)))
                g_loss = g_loss * weight_scale
                g_loss.backward()
                with torch.no_grad():
                    g_grad = [param.grad for param in self.net.parameters()]
    
                self.zero_grad()
                indices2 = permutation2[i:i + self.mini_batch_size]
                batch_x2 = set_x[indices2]
                batch_y2 = set_y[indices2]
                temp_w = torch.exp(scale*self.loss_dual(batch_x2, self.forward(batch_x2, t), batch_y2, self.noise))
                temp_numerator = self.loss_wmse(self.forward(batch_x2, t), batch_y2, temp_w)
                f1_grad = - temp_numerator / g_loss**2
        
                self.zero_grad()
                f2_loss = temp_numerator / g_loss.detach()
                f2_loss.backward()
                with torch.no_grad():
                    f2_grad = [param.grad for param in self.net.parameters()]
                
                with torch.no_grad():
                    grads_squared_new = []
                    for param, g_g, f2_g, square_g in zip(self.net.parameters(), g_grad, f2_grad, grads_squared):
                        grad_current = g_g * f1_grad + f2_g
                        square_g = 0.99 * square_g + 0.01 * torch.square(grad_current)
                        param -= self.lr / (torch.sqrt(square_g) + 1e-8) * grad_current
                        grads_squared_new.append(square_g)
                    grads_squared = grads_squared_new

        set_w = torch.exp(scale*self.loss_dual(set_x, self.forward(set_x, t), set_y, self.noise))
        _, indices_first = torch.sort(set_w[0:self.memories], descending=True)
        _, indices_second = torch.sort(set_w[self.memories:], descending=True)
        indices_combine = torch.cat((indices_first[0:round(self.memories*(1.0-1.0/(self.age+1)))], indices_second[0:round(self.memories/(self.age+1))]))
        self.M = (set_x[indices_combine], set_y[indices_combine])
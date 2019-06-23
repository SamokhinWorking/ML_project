import math
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.parameter import Parameter


NUM_LAYERS = 2
HIDDEN_DIM = 2
LEARNING_RATE = 1e-3
NUM_ITERS=int(1e4)
RANGE=[-100.,100.]
ARITHMETIC_FUNCTIONS = {
    'add': lambda x, y: x + y,
    'sub': lambda x, y: x - y,
    'mul': lambda x, y: x * y,
    'div': lambda x, y: x / y,
}

# The Multilayer Perceptron
class MLP(nn.Module):

    def __init__(self, num_layers, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.num_layers = num_layers
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.activation = nn.ReLU()

        nonlin = True
        if self.activation is None:
            nonlin = False

        layers = []
        for i in range(num_layers - 1):
            layers.extend(
                self._layer(
                    hidden_dim if i > 0 else in_dim,
                    hidden_dim,
                    nonlin,
                )
            )
        layers.extend(self._layer(hidden_dim, out_dim, False))

        self.model = nn.Sequential(*layers)

        # init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(m.bias, -bound, bound)

    def _layer(self, in_dim, out_dim, activation=True):
        if activation:
            return [
                nn.Linear(in_dim, out_dim),
                self.activation,
            ]
        else:
            return [
                nn.Linear(in_dim, out_dim),
            ]

    def forward(self, x):
        out = self.model(x)
        return out





class NeuralAccumulatorCell(nn.Module):

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.W_hat = Parameter(torch.Tensor(out_dim, in_dim))
        self.M_hat = Parameter(torch.Tensor(out_dim, in_dim))
        self.register_parameter('bias', None)

        init.kaiming_uniform_(self.W_hat, a=math.sqrt(5))
        init.kaiming_uniform_(self.M_hat, a=math.sqrt(5))

    def forward(self, input):  
        W = torch.tanh(self.W_hat) * torch.sigmoid(self.M_hat)      

        return F.linear(input, W, self.bias)


class NAC(nn.Module):

    def __init__(self, num_layers, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.num_layers = num_layers
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        layers = []
        for i in range(num_layers):
            layers.append(
                NeuralAccumulatorCell(
                    hidden_dim if i > 0 else in_dim,
                    hidden_dim if i < num_layers - 1 else out_dim,
                )
            )
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        out = self.model(x)
        return out


class NeuralArithmeticLogicUnitCell(nn.Module):
 
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.eps = 1e-10

        self.G = Parameter(torch.Tensor(out_dim, in_dim))
        self.nac = NeuralAccumulatorCell(in_dim, out_dim)
        self.register_parameter('bias', None)

        init.kaiming_uniform_(self.G, a=math.sqrt(5))

    def forward(self, input):
        a = self.nac(input)
        g = torch.sigmoid(F.linear(input, self.G, self.bias))
        add_sub = g * a
        log_input = torch.log(torch.abs(input) + self.eps)
        m = torch.exp(self.nac(log_input))
        mul_div = (1 - g) * m
        y = add_sub + mul_div
        return y


class NALU(nn.Module):
 
    def __init__(self, num_layers, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.num_layers = num_layers
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        layers = []
        for i in range(num_layers):
            layers.append(
                NeuralArithmeticLogicUnitCell(
                    hidden_dim if i > 0 else in_dim,
                    hidden_dim if i < num_layers - 1 else out_dim,
                )
            )
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        out = self.model(x)
        return out


def generate_data(num_train, num_test,fn,support):

    X, y = [], []
    for i in range(num_train + num_test):
        a=random.uniform(support[0],support[1])
        b=random.uniform(support[0],support[1])
        X.append([a, b])
        y.append(fn(a, b))
    X = torch.FloatTensor(X)
    y = torch.FloatTensor(y).unsqueeze_(1)
    indices = list(range(num_train + num_test))
    np.random.shuffle(indices)
    X_train, y_train = X[indices[num_test:]], y[indices[num_test:]]
    X_test, y_test = X[indices[:num_test]], y[indices[:num_test]]
    return X_train, y_train, X_test, y_test


def train(model, optimizer, data, target, num_iters):
    for i in range(num_iters):
        out = model(data)
        loss = F.mse_loss(out, target)
        mea = torch.mean(torch.abs(target - out))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % 1000 == 0:
            print("\t{}/{}: loss: {:.7f} - mea: {:.7f}".format(
                i+1, num_iters, loss.item(), mea.item())
            )


def test(model, data, target):
    with torch.no_grad():
        out = model(data)
        return torch.abs(target - out)


models = [
        MLP(
            num_layers=NUM_LAYERS,
            in_dim=2,
            hidden_dim=HIDDEN_DIM,
            out_dim=1,
        
        ),
        NAC(
            num_layers=NUM_LAYERS,
            in_dim=2,
            hidden_dim=HIDDEN_DIM,
            out_dim=1,
        ),
        NALU(
            num_layers=NUM_LAYERS,
            in_dim=2,
            hidden_dim=HIDDEN_DIM,
            out_dim=1
        ),
        
    ]

results = {}

for fn_str, fn in ARITHMETIC_FUNCTIONS.items():
    results[fn_str] = []
    X_train, y_train, X_test, y_test = generate_data(
        num_train=1000, num_test=200, fn=fn,
        support=RANGE,
    )

    for net in models:
        optim = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)
        train(net, optim, X_train, y_train, NUM_ITERS)
        mse = test(net, X_test, y_test).mean().item()
        results[fn_str].append(mse)








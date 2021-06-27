"""
Can I use a neural network to model a sinusoidal function?
"""
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
torch.manual_seed(0)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# generate data

N = 2000
x = torch.linspace(-2*np.pi, 2*np.pi, N, device=device).reshape(N, 1)
y = torch.sin(x)


# define neural network

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.stack = nn.Sequential(
            nn.Linear(1,200),
            nn.ReLU(),
            nn.Linear(200, 100),
            nn.ReLU(),
            nn.Linear(100, 1),
        )

    def forward(self, x):
        return self.stack(x)

model = NeuralNetwork().to(device)

iterations = 600

fix, ax = plt.subplots(1,2)

learning_rate = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
L = torch.nn.MSELoss(reduction = "sum")

final_loss = 0.0
losses = []
times = []
for t in range(iterations):
    y_hat = model(x)
    loss = L(y_hat, y)
    times.append(t)
    losses.append(loss.item())
    if t % 100 == 99:
        print(f"Loss at time {t}: {loss.item()}")
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

final_loss = loss.item()
print(f"Final loss: {final_loss}")

ax[0].plot(times, losses, label = f"learning rate: {learning_rate}", color = "b")
ax[1].plot(x.cpu().detach().numpy(), y_hat.cpu().detach().numpy(), label = "y_hat", color = "b")
x = np.linspace(-2*np.pi, 2*np.pi, 10000)
ax[1].plot(x, np.sin(x), label = "y", color = "orange")

ax[0].set_xlabel("iteration")
ax[0].set_ylabel("loss")

ax[1].set_xlabel("x")
ax[1].set_ylabel("sin(x)")

ax[0].legend()
ax[1].legend()

plt.savefig("./media/nn_sin.png", dpi = 300)
plt.show()
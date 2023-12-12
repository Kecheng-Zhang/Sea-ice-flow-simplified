import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tqdm import *
from torch import nn
import torch.utils.data.dataset as dataset
import torch.utils.data.dataloader as dataloader


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def update_state(data, dt, f, index):
    '''Update the state of the system
    
    Args:
        data: np.array
        dt: float
        f: function
    
    Returns:
        x_next: float
        v_next: float
    '''
    x = data[0]
    v = data[1]
    t = index * dt
    u = f(x, t)
    x_next = x + v*dt
    v_next = (u-v) * np.abs(u-v) * dt + v
    return np.array([x_next, v_next])


class PhysicsLoss(nn.Module):
    '''Customize PINN loss
    '''
    def __init__(self, dt, f):
        super(PhysicsLoss, self).__init__()
        self.dt = dt
        self.f = f
    
    def forward(self, pred, tar):
        loss1 = (pred[0] - tar[0] - tar[1]*self.dt) ** 2
        u = self.f(tar[0])
        loss2 = (pred[1] - tar[1] - (u - tar[1]) * torch.abs(u - tar[1]) * self.dt) ** 2
        return (loss1 + loss2).sum()
    

class SeaiceDataset(dataset.Dataset):
    def __init__(self, initial, iter, dt, f, resize=1):
        self.data = []
        self.data.append(initial)
        self.dt = dt
        self.f = f
        for i in range(1,iter):
            self.data.append(update_state(self.data[-1], self.dt, self.f, i))
        self.data = np.array(self.data)
        self.data = self.data[::resize]
        self.data_flat = self.data.swapaxes(1,2).reshape(-1,2,1)
        self.data_flat = torch.tensor(self.data_flat, dtype=torch.float32)
        self.data = torch.tensor(self.data, dtype=torch.float32)
        
    def __getitem__(self, index):
        return self.data_flat[index]
    
    def __len__(self):
        return len(self.data_flat)


class SeaiceModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SeaiceModel, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return x
    
    def predict(self, x):
        pred = self.forward(x)
        return pred


def train(model, dataset, f, epochs=3, dt=(10 ** (-3))):
    '''Train the model
    
    Args:
        model: SeaiceModel
        data: SeaiceDataset
        epochs: int
    
    Returns:
    '''
    criterion = PhysicsLoss(dt, f)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    for epoch in range(epochs):
        print("Epoch: ", epoch)
        for data in tqdm(dataset):
            pred = model.predict(data)
            loss = criterion(pred, data)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
    return model


def test(model, dataset):
    '''Test the model
    
    Args:
        model: SeaiceModel
        data: SeaiceDataset
    
    Returns:
        loss: float
    '''
    criterion = nn.MSELoss()
    preds = []
    model.eval()
    loss = 0
    losses = []
    for data in tqdm(dataset):
        pred = model.predict(data)
        preds.append(pred)
        loss = criterion(pred, data)
        losses.append(loss.item())
    
    return np.array(losses).mean() ** (0.5) # return RMSE


def draw_animation(data, gif_name):
    # create animation
    preds = torch.stack(preds).detach().numpy()
    data_np = dataset.data.detach().numpy()
    fig = plt.figure()
    def update(i):
        plt.cla()
        plt.xlim(-1, 7)
        plt.ylim(-1, 1)
        x1 = preds[i][0][0]
        x2 = preds[i][0][1]
        x1_t = data_np[i][0][0]
        x2_t = data_np[i][0][1]
        plt.plot(x1, 0, 'ro', label='Ice 1')
        plt.plot(x2, 0, 'bo', label='Ice 2')
        plt.plot(x1_t, 0, 'r*', label='Ice 1 (true)')
        plt.plot(x2_t, 0, 'b*', label='Ice 2 (true)')
        plt.xlabel('Position')
        plt.title(gif_name+' Time: {:.2f}'.format(i*0.001))
        plt.legend(loc='upper right')
    animation = FuncAnimation(fig, update, frames=range(1, len(preds), 100), interval=1)
    animation.save(gif_name+".gif", fps=100)
    plt.close()
    
    # save the loss graph
    plt.figure()
    plt.plot(range(len(losses)), losses)
    plt.title(gif_name+' Loss')
    plt.xlabel('Time')
    plt.ylabel('Loss')
    plt.savefig(gif_name+"_loss.png")
    plt.close()
    return

def f1(x, t):
    return 0.5 + 0.3*np.sin(np.pi*x + t)

def f2(x, t):
    return 0.2 + 0.5*np.sin(np.pi*x + t)

def f_osc(x):
    return 0.5*np.sin(2*np.pi*x)

def f_rand(x, t):
    a = np.random.random()
    b = np.random.random()
    return a + b * np.sin(np.pi*x + t)

'''
The main program
'''
iterations = 10000 # number of iterations
dt = 10 ** (-3) # time step
epochs = 3 # epoches
initial_data = np.array([[0.3, 0.7], [0, 0]])
train_dataset = SeaiceDataset(initial_data, iterations, dt, f1)

seaice_model = SeaiceModel(2, 10, 2)
if os.path.exists("seaice_model.pt"):
    pass
    #seaice_model.load_state_dict(torch.load("seaice_model.pt"))
else:
    pass
    #seaice_model = train(seaice_model, train_dataset, f1, epochs, dt)
    #torch.save(seaice_model.state_dict(), "seaice_model.pt")

initial_test_data = np.array([[0.1, 0.2], [0.3, 0.4]])
print("initial data:")
print(initial_test_data)
test_data = SeaiceDataset(initial_test_data, iterations//2000, dt, f2)
print("test data:")
print(test_data.data_flat.detach().numpy().reshape(-1,2,2).swapaxes(1,2))
#rmse = test(seaice_model, test_data, "seaice_f1")
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
    x = data[:,0]
    v = data[:,1]
    t = index * dt
    u = f(x, t)
    x_next = x + v*dt
    v_next = (u-v) * np.abs(u-v) * dt + v
    return np.array([x_next, v_next]).T


class PhysicsLoss(nn.Module):
    '''Customize PINN loss
    '''
    def __init__(self, dt, f):
        super(PhysicsLoss, self).__init__()
        self.dt = dt
        self.f = f
    
    def forward(self, pred, tar, index):
        loss1 = (pred[:,0] - tar[:,0] - tar[:,1]*self.dt) ** 2
        u = self.f(tar[:,0], index*self.dt)
        loss2 = (pred[:,1] - tar[:,1] - (u - tar[:,1]) * torch.abs(u - tar[:,1]) * self.dt) ** 2
        return (loss1 + loss2).sum()
    

class SeaiceDataset(dataset.Dataset):
    def __init__(self, initial, iter, dt, f, resize=1):
        self.data = np.zeros((iter,initial.shape[0],initial.shape[1]))
        self.data[0,:,:] = initial[:,:]
        self.dt = dt
        self.f = f
        for i in range(1,iter):
            new_state = update_state(self.data[i-1], self.dt, self.f, i)
            self.data[i,:,:] = new_state[:,:]
        self.data = np.array(self.data)
        self.data = self.data[::resize]
        self.data = torch.tensor(self.data, dtype=torch.float32)
        
    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return len(self.data)


class SeaiceModel(nn.Module):
    '''The model of sea ice
    '''
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
        count = 0
        for data in tqdm(dataset):
            pred = model.predict(data)
            loss = criterion(pred, data, count)
            count += 1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
    return model


def test(model, dataset):
    '''Test the model
    
    Args:
        model: SeaiceModel
        dataset: SeaiceDataset
    
    Returns:
        loss: float
    '''
    criterion = nn.MSELoss()
    preds = np.zeros(dataset.data.numpy().shape)
    print(preds.shape)
    model.eval()
    loss = 0
    losses = np.zeros(dataset.data.numpy().shape[0])
    for i in tqdm(range(len(dataset))):
        data = dataset[i]
        pred = model.predict(data)
        preds[i] = pred.detach().numpy()
        loss = criterion(pred, data)
        losses[i] = loss.item()
    
    rmse = np.array(losses).mean() ** (0.5) # return RMSE
    return preds, losses, rmse


def draw_animation(preds, test, losses, gif_name):
    # create animation
    fig = plt.figure()
    ys = np.zeros(preds.shape[1])
    def update(i):
        plt.cla()
        plt.xlim(-1, 7)
        plt.ylim(-1, 1)
        plt.scatter(test[i,:,0], ys, label='Ground Truth', color='red')
        plt.scatter(preds[i,:,0], ys, label='Prediction', color='blue')
        plt.xlabel('Position')
        plt.title(gif_name+' Time: {:.2f}'.format(i*0.001))
        plt.legend(loc='upper right')
    animation = FuncAnimation(fig, update, frames=range(1, len(preds), 100),interval=1)
    animation.save(gif_name+".gif", fps=120)
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
if __name__ == '__main__':
    iterations = 10000 # number of iterations
    dt = 10 ** (-3) # time step
    epochs = 30 # epoches
    initial_data = np.array([[0.3, 0], [0.7, 0]])
    train_dataset = SeaiceDataset(initial_data, iterations, dt, f1)

    print("Generating model...")
    seaice_model = SeaiceModel(2, 10, 2)
    if os.path.exists("seaice_model.pt"):
        pass
        seaice_model.load_state_dict(torch.load("seaice_model.pt"))
    else:
        pass
        seaice_model = train(seaice_model, train_dataset, f1, epochs, dt)
        torch.save(seaice_model.state_dict(), "seaice_model.pt")
    print('Complete!')

    print("Testing...")
    initial_test_data = np.zeros((16,2))
    for i in range(initial_test_data.shape[0]):
        initial_test_data[i,0] = 1 - i/8
    test_data = SeaiceDataset(initial_test_data, iterations*100, dt/100, f2,resize=100)
    preds, losses, rmse = test(seaice_model, test_data)
    draw_animation(preds, test_data.data.numpy(), losses, "seaice")
    print('RMSE: ', rmse)
    print('Complete!')
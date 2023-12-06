import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tqdm import *
from torch import nn
import torch.utils.data.dataset as dataset
import torch.utils.data.dataloader as dataloader


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def update_state(data, dt, f):
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
    u = f(x)
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
    def __init__(self, initial, iter, dt, f):
        self.data = []
        self.data.append(initial)
        self.dt = dt
        self.f = f
        for i in range(iter):
            self.data.append(update_state(self.data[-1], self.dt, self.f))
        self.data = torch.tensor(self.data, dtype=torch.float32)
        
    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return len(self.data)
    
    def resize(self, const):
        self.data = self.data[::const]
        return self.data


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

def train_WW(model, dataset, f, epochs=3, dt=(10 ** (-3))):
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
        counter = 1
        for data in tqdm(dataset):
            pred = model.predict(data)
            loss = criterion(pred, data) * (counter ** 2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
    return model


def test(model, dataset, gif_name):
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
    
    # create animation
    preds = torch.stack(preds).detach().numpy()
    data_np = dataset.data.detach().numpy()
    print(preds.shape)
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
    
    return loss.item()

def f1(x):
    return 0.5 + 0.3*np.sin(2*np.pi*x)

def f2(x):
    return 0.2 + 0.5*np.sin(2*np.pi*x)

'''
The main program
'''

def q1():
    print("Start Q1!")
    iterations = 10000 # number of iterations
    dt = 10 ** (-3) # time step
    epochs = 30 # epoches
    initial_data = np.array([[0.3, 0.7], [0, 0]])
    seaice_dataset = SeaiceDataset(initial_data, iterations, dt, f1)
    # Start training
    seaice_model = SeaiceModel(2, 10, 2)
    seaice_model = train(seaice_model, seaice_dataset, f1, epochs=epochs, dt=dt)
    seaice_model_ww = SeaiceModel(2, 10, 2)
    seaice_model_ww = train_WW(seaice_model_ww, seaice_dataset, f1, epochs=epochs, dt=dt)
    train_loss = test(seaice_model, seaice_dataset, 'Train')
    print("Train MSE loss: ", train_loss)
    train_loss_ww = test(seaice_model_ww, seaice_dataset, 'Train_WW')
    print("Train with Weights MSE loss: ", train_loss_ww)
    print("Training finished!")

    print("Testing...")
    initial_test = np.array([[0.42, 0.87], [-0.1, 0.05]])
    dataset_test = SeaiceDataset(initial_test, iterations, dt, f1)
    test_loss = test(seaice_model, dataset_test, 'Test')
    print("Test MSE Loss: ", test_loss)
    test_loss_ww = test(seaice_model_ww, dataset_test, 'Test_WW')
    print("Test with Weights MSE Loss: ", test_loss_ww)
    print("Done!")
    
    
def q2():
    print("Start Q2!")
    iterations = 10000 # number of iterations
    dt = 10 ** (-3) # time step
    epochs = 30 # epoches
    initial_data = np.array([[0.3, 0.7], [0, 0]])
    seaice_dataset1 = SeaiceDataset(initial_data, iterations, dt, f1)
    # Get the model
    seaice_model = SeaiceModel(2, 10, 2)
    seaice_model = train(seaice_model, seaice_dataset1, f1, epochs=epochs, dt=dt)
    # test the model
    initial_test = np.array([[0.42, 0.87], [-0.1, 0.05]])
    seaice_dataset_test1 = SeaiceDataset(initial_test, iterations, dt, f1)
    
    seaice_dataset_test2 = SeaiceDataset(initial_test, iterations, dt, f2)
    test_loss1 = test(seaice_model, seaice_dataset_test1, 'Test_normal_ocean_field')
    test_loss2 = test(seaice_model, seaice_dataset_test2, 'Test_differnt_ocean_field')
    print("Test MSE Loss: ", test_loss1)
    print("Test MSE Loss: ", test_loss2)
    print("Done!")
    
q2()
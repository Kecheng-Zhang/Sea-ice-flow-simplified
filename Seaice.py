import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tqdm import *
from torch import nn
import torch.utils.data.dataset as dataset
import torch.utils.data.dataloader as dataloader


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def update_state(data, dt):
    '''Update the state of the system
    
    Args:
        data: np.array
        dt: float
    
    Returns:
        x_next: float
        v_next: float
    '''
    x = data[0]
    v = data[1]
    u = 0.5 + 0.3*np.sin(2*np.pi*x)
    x_next = x + v*dt
    v_next = (u-v) * np.abs(u-v) * dt + v
    return np.array([x_next, v_next])


class PhysicsLoss(nn.Module):
    '''Customize PINN loss
    '''
    def __init__(self, dt):
        super(PhysicsLoss, self).__init__()
        self.dt = dt
    
    def forward(self, pred, tar):
        loss = (pred[0] - tar[0] - tar[1]*dt) ** 2
        return loss.sum()
    

class SeaiceDataset(dataset.Dataset):
    def __init__(self, initial, iter, dt):
        self.data = []
        self.data.append(initial)
        for i in range(iter):
            self.data.append(update_state(self.data[-1], dt))
        self.data = torch.tensor(self.data, dtype=torch.float32)
        
    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return len(self.data)


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


def train(model, dataset, epochs=3, dt=(10 ** (-3))):
    '''Train the model
    
    Args:
        model: SeaiceModel
        data: SeaiceDataset
        epochs: int
    
    Returns:
    '''
    criterion = PhysicsLoss(dt)
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
    for data in tqdm(dataset):
        pred = model.predict(data)
        preds.append(pred)
        loss += criterion(pred, data)
    
    preds = torch.stack(preds).detach().numpy()
    data_np = dataset.data.detach().numpy()
    print(preds.shape)
    # create animation
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
        plt.legend(loc='upper right')
    animation1 = FuncAnimation(fig, update, frames=range(1, len(preds), 100), interval=1)
    animation1.save(gif_name, fps=100)
    plt.close()
    return loss.item()
    

'''
The main program
'''
print("Start!")
iterations = 10000 # number of iterations
dt = 10 ** (-3) # time step
epochs = 10 # epoches
initial_data = np.array([[0.3, 0.7], [0, 0]])
seaice_dataset = SeaiceDataset(initial_data, iterations, dt)
seaice_model = SeaiceModel(2, 10, 2)
seaice_model = train(seaice_model, seaice_dataset, epochs=epochs, dt=dt)
train_loss = test(seaice_model, seaice_dataset, 'train.gif')
print("Train MSE loss: ", train_loss)
print("Training finished!")

print("Testing...")
initial_test = np.array([[0.42, 0.87], [-0.1, 0.05]])
dataset_test = SeaiceDataset(initial_test, iterations, dt)
test_loss = test(seaice_model, dataset_test, 'test.gif')
print("Test MSE Loss: ", test_loss)
print("Done!")
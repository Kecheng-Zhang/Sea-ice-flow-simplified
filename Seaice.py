import torch
import numpy as np
from torch import nn
import torch.utils.data.dataset as dataset
import torch.utils.data.dataloader as dataloader


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_next_time_data(x, v, dt):
    '''Get the x(tj+1) and v(tj+1) from x(tj) and v(tj)
    Args:
        x: float
        v: float
        dt: float
    Returns:
        x_next: float
        v_next: float
    '''
    u = 0.5 + 0.3*np.sin(2*np.pi*x)
    x_next = x + v*dt
    v_next = (u-v) * np.abs(u-v) * dt + v
    return x_next, v_next
    

class SeaiceDataset(dataset.Dataset):
    def __init__(self, data):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.d1 = self.data[0, :]
        self.d2 = self.data[1, :]
        
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
        x = torch.relu(self.linear1(x))
        x = self.linear2(x)
        return x
    
    def predict(self, x):
        pred = self.forward(x)
        return pred


def train(model, dataset, iters=10000, dt=(10 ** (-3))):
    '''Train the model
    
    Args:
        model: SeaiceModel
        data: SeaiceDataset
        epochs: int
    
    Returns:
    '''
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    for iter in range(iters):
        
        for data in dataset:
            pred = model.predict(data)
            
            x, v = data
            x_next, v_next = get_next_time_data(x, v, dt)
            data_next = torch.tensor([x_next, v_next])
            
            loss = criterion(pred, data_next)
            loss += pred[0] - data_next[0] - data_next[1]
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
    pass


'''
The main program
'''
if __name__ == '__main__':
    print("Start!")
    initial_data = np.array([[0.3, 0.7], [0, 0]])
    seaice_dataset = SeaiceDataset(initial_data)
    seaice_dataloader = dataloader.DataLoader(seaice_dataset, batch_size=1, shuffle=True)
    seaice_model = SeaiceModel(2, 8, 2)
    
    train(seaice_model, seaice_dataset)
    print("Done!")
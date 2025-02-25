import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseModel(nn.Module):
    def __init__(self, input_size=10, hidden_size=256, output_size=10):
        super(BaseModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.extra_fc = nn.Linear(hidden_size, hidden_size)  # Ensure this layer exists
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.extra_fc(x))  # Ensure this layer is used
        x = self.fc2(x)
        return x

def get_phone_model():
    return BaseModel()

def get_pc_model():
    return BaseModel()

def get_server_model():
    return BaseModel()

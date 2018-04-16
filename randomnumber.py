import secrets
import numpy as np
from tqdm import tqdm
from IPython.display import Image
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class LSTMCat(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTMCat, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim)
        self.lin = nn.Linear(hidden_dim, output_dim)
        self.sm = nn.Softmax(dim=0)
        self.hidden_dim = hidden_dim
        self.hidden = self.init_hidden()
        self = self.apply(LSTMCat.init_weights)
        
    def init_hidden(self):
        return(
            autograd.Variable(torch.zeros(1, 1, self.hidden_dim)),
            autograd.Variable(torch.zeros(1, 1, self.hidden_dim))
        )

    def forward(self, x):
        out, self.hidden = self.lstm(x, self.hidden)
        out = self.sm(self.lin(out))
        return(out)
    
    @staticmethod
    def init_weights(m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            m.weight.data.normal_(0.0, 0.2)

    
class Helper():
    
    def __init__(self, verbose=False):
        self.alph = "абвгдеёжзийклмнопрстуфхцчшщьыъэюя(:!-.,) "
        np.random.seed(42)
        torch.manual_seed(42)
        self.message = "покушай и спатки:-)"
        self.parameters = {
            "learning_rate": 0.01,
            "num_epochs": 100000
        }
        self.verbose = verbose
        self.table = {}
        for i,a in enumerate(self.alph):
            self.table[a] = i
        self.get_random_number()
        self.get_Y()
    
    def get_random_number(self):
        """Pseudorandom number"""
        string = bin(secrets.randbits(761))[2:]
        if len(string) < 779:
            string += "0"*(779-len(string))
        if len(string) > 779:
            string = string[0:779]
        self.X = np.array([int(a) for a in string])
        self.number = string
    
    def get_Y(self):
        """Y's for the model"""
        Y = []
        for a in self.message:
            Y.append(self.table[a])
        self.Y = np.array(Y)
        
    def init_model(self):
        """Just a simple LSTM network"""
        self.lstm = LSTMCat(len(self.alph), 32, len(self.alph))
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.lstm.parameters(), lr=self.parameters["learning_rate"])
        
    def predict(self, x):
        self.lstm.hidden = self.lstm.init_hidden()
        X = autograd.Variable(torch.from_numpy(x).type(torch.FloatTensor))
        self.lstm.hidden = self.lstm.init_hidden()
        return(self.lstm(X))
    
    def train(self):
        for a in tqdm(range(self.parameters["num_epochs"])):
            X = autograd.Variable(
                torch.from_numpy(
                    self.X.reshape((19, 1, len(self.alph)))
                ).type(torch.FloatTensor)
            )
            Y = autograd.Variable(torch.from_numpy(self.Y).type(torch.LongTensor))
            self.lstm.zero_grad()
            self.lstm.hidden = self.lstm.init_hidden()
            predictions = self.lstm(X).view(19,41)
            msg = "".join([self.alph[b] for b in np.argmax(predictions.data.numpy(), 1)])
            if self.verbose:
                print(msg)
            loss = self.loss_function(predictions, Y)
            loss.backward()
            self.optimizer.step()
        return(msg)
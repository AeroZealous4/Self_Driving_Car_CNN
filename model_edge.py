#Model Architecture for Self driving car

import torch.nn as nn

class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        self.linear_layers = nn.Sequential(           
            nn.Linear(in_features=24, out_features=100),  #number may differ
            nn.ELU(),
            #nn.Linear(in_features=100, out_features=50),
            #nn.ELU(),
            nn.Linear(in_features=100, out_features=10),
            nn.ELU(),
            #nn.Linear(in_features=10, out_features=10),
            nn.Linear(in_features=10, out_features=1)
        )
        
    def forward(self, input):  
        #input = input.view(input.size(0), 3, 70, 320)
        # output = self.conv_layers(input)
        output = input.view(input.size(0), -1)
        output = self.linear_layers(output)
        return output
class CNN3(nn.Module):

    def __init__(self):
        super(CNN3, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 1, 5, stride=2),#160 or 70
            nn.ELU()
            #nn.Dropout(0.25)
        )
        self.linear_layers = nn.Sequential(             
            nn.Linear(in_features=33*158, out_features=10),  #number may differ
            nn.ELU(),
            #nn.Linear(in_features=100, out_features=50),
            #nn.ELU(),
            # nn.Linear(in_features=100, out_features=10),
            # nn.ELU(),
            #nn.Linear(in_features=10, out_features=10),
            nn.Linear(in_features=10, out_features=1)
        )
        
    def forward(self, input):  
        #input = input.view(input.size(0), 3, 70, 320)
        # print(input.shape)
        output = self.conv_layers(input)
        # print(input.shape)
        output = output.view(output.size(0), -1)
        output = self.linear_layers(output)
        return output

class CNN5(nn.Module):

    def __init__(self):
        super(CNN5, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 5, 5, stride=2),#160 or 70
            nn.ELU(),
            nn.Conv2d(5,5,5,stride=2),
            nn.ELU(),
            nn.Conv2d(5,1,5,stride=2),
            nn.ELU()

            #nn.Dropout(0.25)
        )
        self.linear_layers = nn.Sequential(             
            nn.Linear(in_features=38*18, out_features=10),  #number may differ
            nn.ELU(),
            #nn.Linear(in_features=100, out_features=50),
            #nn.ELU(),
            # nn.Linear(in_features=100, out_features=10),
            # nn.ELU(),
            #nn.Linear(in_features=10, out_features=10),
            nn.Linear(in_features=10, out_features=1)
        )
        
    def forward(self, input):  
        #input = input.view(input.size(0), 3, 70, 320)
        # print(input.shape)
        output = self.conv_layers(input)
        # print(input.shape)
        output = output.view(output.size(0), -1)
        output = self.linear_layers(output)
        return output

        

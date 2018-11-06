import torch
import numpy as np
import os
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Mask Vector and Hint Vector Generation
def sample_M(m, n, p):
    A = np.random.uniform(0., 1., size = [m, n])
    B = A > p
    C = 1.*B
    return C
    
# 2. Plot (4 x 4 subfigures)
def plot(samples):
    fig = plt.figure(figsize = (5,5))
    gs = gridspec.GridSpec(5,5)
    gs.update(wspace=0.05, hspace=0.05)
    
    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28,28), cmap='Greys_r')
        
    return fig

#%% 3. Others
# Random sample generator for Z
def sample_Z(m, n):
    return np.random.uniform(0., 1., size = [m, n])


def sample_idx(m, n):
    A = np.random.permutation(m)
    idx = A[:n]
    return idx


class NetD(torch.nn.Module):
    def __init__(self):
        super(NetD, self).__init__()
        self.fc1 = torch.nn.Linear(Dim*2, 256)
        self.fc2 = torch.nn.Linear(256, 128)
        self.fc3 = torch.nn.Linear(128, Dim)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.init_weight()
    
    
    def init_weight(self):
        layers = [self.fc1, self.fc2, self.fc3]
        [torch.nn.init.xavier_normal_(layer.weight) for layer in layers]
        
        
    def forward(self, x, m, g, h):
        """Eq(3)"""
        inp = m * x + (1-m) * g 
        inp = torch.cat((inp, h), dim=1)
        out = self.relu(self.fc1(inp))
        out = self.relu(self.fc2(out))
#         out = self.sigmoid(self.fc3(out)) # [0,1] Probability Output
        out = self.fc3(out)
        
        return out    

"""
Eq(2)
"""
class NetG(torch.nn.Module):
    def __init__(self):
        super(NetG, self).__init__()
        self.fc1 = torch.nn.Linear(Dim*2, 256)
        self.fc2 = torch.nn.Linear(256, 128)
        self.fc3 = torch.nn.Linear(128, Dim)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.init_weight()
    
    def init_weight(self):
        layers = [self.fc1, self.fc2, self.fc3]
        [torch.nn.init.xavier_normal_(layer.weight) for layer in layers]
        
        
    def forward(self, x, z, m):
        inp = m * x + (1-m) * z
        inp = torch.cat((inp, m), dim=1)
        out = self.relu(self.fc1(inp))
        out = self.relu(self.fc2(out))
        out = self.sigmoid(self.fc3(out)) # [0,1] Probability Output
#         out = self.fc3(out)
        
        return out 


# 1. Mini batch size
mb_size = 128
# 2. Missing rate
p_miss = 0.5
# 3. Hint rate
p_hint = 0.9
# 4. Loss Hyperparameters
alpha = 10
# 5. Imput Dim (Fixed)
Dim = 784
# 6. No
Train_No = 55000
Test_No = 10000

mnist = input_data.read_data_sets('../../MNIST_data', one_hot = True)

trainX, _ = mnist.train.next_batch(Train_No) 
testX, _  = mnist.test.next_batch(Test_No) 

trainM = sample_M(Train_No, Dim, p_miss)
testM = sample_M(Test_No, Dim, p_miss)


netD = NetD()
netG = NetG()


optimD = torch.optim.Adam(netD.parameters(), lr=0.001)
optimG = torch.optim.Adam(netG.parameters(), lr=0.001)


# Output Initialization
if not os.path.exists('Multiple_Impute_out1/'):
    os.makedirs('Multiple_Impute_out1/')


bce_loss = torch.nn.BCEWithLogitsLoss(reduction="elementwise_mean")
mse_loss = torch.nn.MSELoss(reduction="elementwise_mean")

i = 1
#%% Start Iterations
for it in range(10000): 
    #%% Inputs
    mb_idx = sample_idx(Train_No, mb_size)
    X_mb = trainX[mb_idx,:]  
    Z_mb = sample_Z(mb_size, Dim) 
    
    M_mb = trainM[mb_idx,:]  
    H_mb1 = sample_M(mb_size, Dim, 1-p_hint)
    H_mb = M_mb * H_mb1 + 0.5*(1-H_mb1)
    
    New_X_mb = M_mb * X_mb + (1-M_mb) * Z_mb  # Missing Data Introduce
    
    X_mb = torch.tensor(X_mb).float()
    New_X_mb = torch.tensor(New_X_mb).float()
    Z_mb = torch.tensor(Z_mb).float()
    M_mb = torch.tensor(M_mb).float()
    H_mb = torch.tensor(H_mb).float()
    
    # Train D
    G_sample = netG(X_mb, New_X_mb, M_mb)
    D_prob = netD(X_mb, M_mb, G_sample, H_mb)
    D_loss = bce_loss(D_prob, M_mb)
    
    D_loss.backward()
    optimD.step()
    optimD.zero_grad()
    
    # Train G
    G_sample = netG(X_mb, New_X_mb, M_mb)
    D_prob = netD(X_mb, M_mb, G_sample, H_mb)
    D_prob.detach_()
    G_loss1 = ((1 - M_mb) * (torch.sigmoid(D_prob)+1e-8).log()).mean()/(1-M_mb).sum()
    G_mse_loss = mse_loss(M_mb*X_mb, M_mb*G_sample) / M_mb.sum()
    G_loss = G_loss1 + alpha*G_mse_loss
    
    G_loss.backward()
    optimG.step()
    optimG.zero_grad()
    
    G_mse_test = mse_loss((1-M_mb)*X_mb, (1-M_mb)*G_sample) / (1-M_mb).sum()
    
  #%% Output figure
    if it % 100 == 0:
      
        mb_idx = sample_idx(Test_No, 5)
        X_mb = testX[mb_idx,:]
        M_mb = testM[mb_idx,:]  
        Z_mb = sample_Z(5, Dim) 
        
        New_X_mb = M_mb * X_mb + (1-M_mb) * Z_mb
        
        X_mb = torch.tensor(X_mb).float()
        New_X_mb = torch.tensor(New_X_mb).float()
        Z_mb = torch.tensor(Z_mb).float()
        M_mb = torch.tensor(M_mb).float()
        
        samples1 = X_mb                
        samples5 = M_mb * X_mb + (1-M_mb) * Z_mb
        
        samples2 = netG(X_mb, New_X_mb, M_mb)
        samples2 = M_mb * X_mb + (1-M_mb) * samples2        
        
        Z_mb = torch.Tensor(sample_Z(5, Dim)).float()
        New_X_mb = M_mb * X_mb + (1-M_mb) * Z_mb  
        
        samples3 =netG(X_mb, New_X_mb, M_mb)
        samples3 = M_mb * X_mb + (1-M_mb) * samples3     
        
        Z_mb = torch.tensor(sample_Z(5, Dim)).float()
        New_X_mb = M_mb * X_mb + (1-M_mb) * Z_mb       
        samples4 = netG(X_mb, New_X_mb, M_mb)
        samples4 = M_mb * X_mb + (1-M_mb) * samples4     
        
        
        samples = np.vstack([samples5.detach().data, samples2.detach().data, samples3.detach().data,
                             samples4.detach().data, samples1.detach().data])          
        
        fig = plot(samples)
        plt.savefig('Multiple_Impute_out1/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
        i += 1
        plt.close(fig)
        
        
    #%% Intermediate Losses
    if it % 100 == 0:
        print('Iter: {}'.format(it))
        print('D_loss: {:.4}'.format(D_loss))
        print('Train_loss: {:.4}'.format(G_mse_loss))
        print('Test_loss: {:.4}'.format(G_mse_test))
        print()
    
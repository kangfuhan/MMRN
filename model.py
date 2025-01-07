import torch
import torch.nn.functional as F
from torch import nn

from config import args

class Conv3d(nn.Module):
    def __init__(self, channels_in, channels_out, kernel_size = 3, stride = 1, padding = 1):
        super(Conv3d, self).__init__()
        self.conv = nn.Conv3d(channels_in,channels_out, kernel_size = kernel_size, stride = stride, padding = padding)
        self.bn = nn.BatchNorm3d(channels_out)
        self.activate = nn.ReLU(inplace = True)
    
    def forward(self, input):

        return self.activate(self.bn(self.conv(input)))

class Encoder(nn.Module):

    def __init__(self):
        super(Encoder, self).__init__()
        
        self.encoder = nn.Sequential(
            Conv3d(1              , args.Filters[0]),
            Conv3d(args.Filters[0], args.Filters[0]),
            nn.MaxPool3d((2, 2, 2)),
            Conv3d(args.Filters[0], args.Filters[1]),
            Conv3d(args.Filters[1], args.Filters[1]),
            nn.MaxPool3d((2, 2, 2)),
            Conv3d(args.Filters[1], args.Filters[2]),
            nn.MaxPool3d((2, 2, 2)),
            Conv3d(args.Filters[2], args.Filters[3]),
            nn.MaxPool3d((2, 2, 2)),
            # nn.Dropout3d(args.dropout),
            Conv3d(args.Filters[3], args.Filters[4]),
            nn.MaxPool3d((2, 2, 2)),
            # nn.Dropout3d(args.dropout),
            Conv3d(args.Filters[4], args.Filters[5]),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten(), 
            # nn.Dropout(args.dropout),   
        )

    def forward(self, inputs):

        return self.encoder(inputs)
    
class Disentanglement(nn.Module):

    def __init__(self):
        super(Disentanglement, self).__init__()
        
        self.L1 = nn.Linear(args.Filters[5], args.Latent)
        self.L2 = nn.Linear(args.Filters[5], args.Latent)
    
    def forward(self, inputs):
        
        feat_cls  = F.relu(self.L1(inputs))
        feat_meta = F.relu(self.L2(inputs))

        return feat_cls, feat_meta
    
class Classifier(nn.Module):

    def __init__(self):
        super(Classifier, self).__init__()
        
        self.cls = nn.Linear(args.Latent, args.task_num_classes)  

    def forward(self, inputs):

        return self.cls(inputs)
    
class Projection(nn.Module):

    def __init__(self):
        super(Projection, self).__init__()
        
        self.proj = nn.Sequential(
            nn.Linear(args.Latent, args.Latent),
            nn.ReLU()
            )  

    def forward(self, inputs):

        return self.proj(inputs)
    
class Discriminator(nn.Module):

    def __init__(self, input_dim = args.Latent):
        super(Discriminator, self).__init__()
        
        self.H = nn.Sequential(
            nn.Linear(input_dim, args.Latent//2), 
            nn.LeakyReLU(0.2),
            nn.Linear(args.Latent//2, args.Latent//4), 
            nn.LeakyReLU(0.2)    
        )
        
        self.D = nn.Linear(args.Latent//4, 1)
        self.Discrete = nn.Linear(args.Latent//4, 2)  #Gender
        self.Continus = nn.Linear(args.Latent//4, 2)  #Age + Education

    def forward(self, inputs):

        x = self.H(inputs)
        return torch.sigmoid(self.D(x)), self.Discrete(x), self.Continus(x)
    
class Generator(nn.Module):

    def __init__(self, input_dim = 64):
        super(Generator, self).__init__()
        
        self.G = nn.Sequential(
            nn.Linear(input_dim, args.Latent//2),
            nn.BatchNorm1d(args.Latent//2),    
            nn.LeakyReLU(0.2),   
            nn.Linear(args.Latent//2, args.Latent),
            nn.BatchNorm1d(args.Latent),  
            nn.LeakyReLU(0.2)   
        )

    def forward(self, inputs):

        return self.G(inputs)
    
class Reconstruction(nn.Module):

    def __init__(self, input_dim = args.Latent*2, output_dim = args.Filters[-1]):
        super(Reconstruction, self).__init__()
        
        self.R = nn.Sequential(
            nn.Linear(input_dim, args.Latent),  
            nn.ReLU(),
            # nn.Dropout(args.dropout),   
            nn.Linear(args.Latent, output_dim),   
        )

    def forward(self, inputs):

        return self.R(inputs)

class CLUBSample(nn.Module):  # Sampled version of the CLUB estimator
    """
    Code from https://github.com/Linear95/CLUB
    CLUB: A Contrastive Log-ratio Upper Bound of Mutual Information, ICML 2020
    """
    def __init__(self, x_dim = args.Latent, y_dim = args.Latent, hidden_size = args.Latent//4):
        super(CLUBSample, self).__init__()
        self.p_mu = nn.Sequential(nn.Linear(x_dim, hidden_size),
                                       nn.ReLU(),
                                       nn.Linear(hidden_size, y_dim))

        self.p_logvar = nn.Sequential(nn.Linear(x_dim, hidden_size),
                                       nn.ReLU(),
                                       nn.Linear(hidden_size, y_dim),
                                       nn.Sigmoid()) # The original activation function is nn.Tanh(), please see https://github.com/Linear95/CLUB/issues/12#issuecomment-897361330

    def get_mu_logvar(self, x_samples):
        mu = self.p_mu(x_samples)
        logvar = self.p_logvar(x_samples)
        return mu, logvar
     
        
    def loglikeli(self, x_samples, y_samples):
        mu, logvar = self.get_mu_logvar(x_samples)
        return (-(mu - y_samples)**2 /logvar.exp()-logvar).sum(dim=1).mean(dim=0)
    

    def forward(self, x_samples, y_samples):
        mu, logvar = self.get_mu_logvar(x_samples)
        
        sample_size = x_samples.shape[0]
        #random_index = torch.randint(sample_size, (sample_size,)).long()
        random_index = torch.randperm(sample_size).long()
        
        positive = - (mu - y_samples)**2 / logvar.exp()
        negative = - (mu - y_samples[random_index])**2 / logvar.exp()
        upper_bound = (positive.sum(dim = -1) - negative.sum(dim = -1)).mean()
        return upper_bound/2.

    def learning_loss(self, x_samples, y_samples):
        return - self.loglikeli(x_samples, y_samples)
  

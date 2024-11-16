import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import dropout_edge,mask_feature

class Conv(torch.nn.Module):
    def __init__(self,input_dim,hidden_dim,proj_dim,activation,num_layers,method=None,drop_out=0.0):
        super(Conv,self).__init__()
        self.activation = activation
        self.layers = torch.nn.ModuleList()
        self.drop_out = drop_out
        self.layers.append(GCNConv(input_dim,hidden_dim))
        for _ in range(num_layers - 1):
            self.layers.append(GCNConv(hidden_dim,hidden_dim))
        
        self.projection_head = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim,proj_dim),
            torch.nn.PReLU(),
            torch.nn.Dropout(drop_out),
            torch.nn.Linear(proj_dim,proj_dim)
        )  
        
    def forward(self,x,edge_index):
        z = x
        for conv in self.layers:
            z = conv(z,edge_index)
            z = self.activation(z)
            z = F.dropout(z,p=self.drop_out,training=self.training)

        return z,self.projection_head(z)

class Online(torch.nn.Module):
    def __init__(self,online_encoder,target_encoder,hidden_dim,slsp_adj,num_hop,momentum):
        super(Online,self).__init__()
        self.online_encoder = online_encoder
        self.target_encoder = target_encoder
        self.slsp_adj = slsp_adj
        self.num_hop = num_hop
        self.momentum = momentum
        self.predictor = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.PReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim)
        )
        
    def update_target_encoder(self):
        
        for p, new_p in zip(self.target_encoder.parameters(), self.online_encoder.parameters()):
            next_p = self.momentum * p.data + (1 - self.momentum) * new_p.data
            p.data = next_p
            
    def forward(self, x, edge_index):
        or_embeds, pr_embeds = self.embed(x,edge_index,self.slsp_adj,self.num_hop)
        h = or_embeds + pr_embeds
        h_pred = self.predictor(h)
        with torch.no_grad():
               h_target,_ = self.target_encoder(x,edge_index)
              
        return h,h_pred,h_target
       
    def get_loss(self,z1,z2):
        z1 = F.normalize(z1, dim=-1, p=2)
        z2 = F.normalize(z2, dim=-1, p=2)
        
        loss = (z1 * z2).sum(dim=-1)
        return -loss.mean()
    
    def embed(self, seq, edge_index, adj, Globalhop=10):
        h_1,_ = self.online_encoder(seq,edge_index)
        h_2 = h_1.clone()
        for _ in range(Globalhop):
            h_2 = adj @ h_2
        return h_1, h_2
    
    
class Target(torch.nn.Module):
    def __init__(self,target_encoder):
        super(Target,self).__init__()
        self.target_encoder = target_encoder
        
    def forward(self,x,edge_index):
        h_target,_ = self.target_encoder(x,edge_index)
        return h_target
    
    def get_loss(self,z):
        z = F.normalize(z,dim=-1, p=2)
        return -(z - z.mean(dim=0)).pow(2).sum(1).mean()



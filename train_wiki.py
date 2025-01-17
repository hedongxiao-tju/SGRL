import numpy as np
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, CitationFull, WikiCS, Amazon, Coauthor, WebKB, Actor, WikipediaNetwork
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, ShuffleSplit, train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import OneHotEncoder, normalize
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import to_undirected
from data import load_dataset
from models import Conv, Online, Target
import torch
import random
import torch_geometric
import copy
import time
from tqdm import tqdm
from torch_geometric.utils import train_test_split_edges
import warnings
import argparse
import sys
import os

def adj_norm(adj_t):
    deg = torch.sparse.sum(adj_t, dim=1).to_dense()
    deg_inv_sqrt = deg.pow(-0.5)
    adj_t = adj_t * deg_inv_sqrt.view(1, -1)
    adj_t = adj_t * deg_inv_sqrt.view(-1, 1)
    return adj_t

import numpy as np
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, ShuffleSplit, train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import OneHotEncoder, normalize

def fit_logistic_regression_preset_splits(X, y, train_masks, val_masks, test_mask):
    # transfrom targets to one-hot vector
    one_hot_encoder = OneHotEncoder(categories='auto')
    y = one_hot_encoder.fit_transform(y.reshape(-1, 1)).astype(bool)

    # normalize x
    X = normalize(X, norm='l2')

    accuracies = []
    for split_id in range(train_masks.shape[1]):
        # get train/val/test masks
        train_mask, val_mask = train_masks[:, split_id], val_masks[:, split_id]

        # make custom cv
        X_train, y_train = X[train_mask], y[train_mask]
        X_val, y_val = X[val_mask], y[val_mask]
        X_test, y_test = X[test_mask], y[test_mask]

        # grid search with one-vs-rest classifiers
        best_test_acc, best_acc = 0, 0
        for c in 2.0 ** np.arange(-10, 11):
            clf = OneVsRestClassifier(LogisticRegression(solver='liblinear', C=c))
            clf.fit(X_train, y_train)

            y_pred = clf.predict_proba(X_val)
            y_pred = np.argmax(y_pred, axis=1)
            y_pred = one_hot_encoder.transform(y_pred.reshape(-1, 1)).astype(bool)
            val_acc = metrics.accuracy_score(y_val, y_pred)
            if val_acc > best_acc:
                best_acc = val_acc
                y_pred = clf.predict_proba(X_test)
                y_pred = np.argmax(y_pred, axis=1)
                y_pred = one_hot_encoder.transform(y_pred.reshape(-1, 1)).astype(bool)
                best_test_acc = metrics.accuracy_score(y_test, y_pred)

        accuracies.append(best_test_acc)
    return accuracies

def get_wiki_cs(root, transform=NormalizeFeatures()):
    dataset = WikiCS(root, transform=transform)
    data = dataset[0]
    std, mean = torch.std_mean(data.x, dim=0, unbiased=False)
    data.x = (data.x - mean) / std
    data.edge_index = to_undirected(data.edge_index)
    return [data], np.array(data.train_mask), np.array(data.val_mask), np.array(data.test_mask)



def train_online(online,optimizer,data):
    online.train()
    h,h_pred,h_target = online(data.x,data.edge_index)
    loss = online.get_loss(h_pred,h_target.detach())
    loss.backward()
    optimizer.step()
    online.update_target_encoder()
    return loss.item()

def train_target(target,optimizier,data):
    target.train()
    h_target = target(data.x,data.edge_index)
    loss = target.get_loss(h_target)
    loss.backward()
    optimizier.step()
    return loss.item()


def run(args):
    
    log_dir = args.log_dir
    path = args.data_dir
    with open(log_dir, 'a') as f:
        f.write(str(args))
        f.write('\n\n\n')
    trials = args.trials
    torch_geometric.seed.seed_everything(args.seed)
    seed = args.seed
    
    for trial in range(trials):
        e1_lr = args.e1_lr
        e2_lr = args.e2_lr
        weight_decay = args.weight_decay
        hidden_dim = args.hidden_dim
        activation = torch.nn.PReLU()
        num_layers = args.num_layers
        num_epochs = args.num_epochs
        momentum = args.momentum
        dataset_name = args.dataset_name
        
        dataset, train_masks, val_masks, test_masks = get_wiki_cs(args.data_dir)
        data = dataset[0]
        device = torch.device('cuda')
        data = data.to(device)
        
        num_hop = args.num_hop
        nb_nodes = data.x.size()[0]
        
        self_loop_for_adj = torch.Tensor([i for i in range(nb_nodes)]).unsqueeze(0)
        self_loop_for_adj = torch.concat([self_loop_for_adj, self_loop_for_adj], dim=0)
        slsp_adj = torch.concat([data.edge_index.to('cpu'), self_loop_for_adj], dim=1)
        slsp_adj = torch.sparse.FloatTensor(slsp_adj.long(), torch.ones(slsp_adj.size()[1]),
                                            torch.Size([nb_nodes, nb_nodes]))
        
        slsp_adj = adj_norm(slsp_adj).to(device)
        

        
        online_conv = Conv(data.x.size()[1],hidden_dim,hidden_dim,activation,num_layers).to(device)
        target_conv = Conv(data.x.size()[1],hidden_dim,hidden_dim,activation,num_layers).to(device)
        
        online_model = Online(online_conv,target_conv,hidden_dim,slsp_adj,num_hop,momentum).to(device)
        target_model = Target(target_conv).to(device)
        
        online_optimizer = torch.optim.Adam(online_model.parameters(),lr=e1_lr)
        target_optimizer = torch.optim.Adam(target_model.parameters(),lr=e2_lr)

        best_online_loss = 1e9
        best_target_loss = 1e9

        tag = dataset_name + '_' + str(time.time())
        with tqdm(total = num_epochs,desc='(T)') as pbar:
            for epoch in range(num_epochs):
                online_optimizer.zero_grad()
                target_optimizer.zero_grad()
                
                online_loss = train_online(online_model,online_optimizer,data)                   
                if online_loss < best_online_loss:
                    best_online_loss = online_loss
                    torch.save(online_model.state_dict(), 'pkl/pkl_online/best_online' + tag + '.pkl')
                    
                target_loss = train_target(target_model,target_optimizer,data)
                
                if target_loss < best_target_loss:
                    best_target_loss_loss = target_loss
                    torch.save(target_model.state_dict(), 'pkl/pkl_target/best_target' + tag + '.pkl')
                    target_model.load_state_dict(torch.load('pkl/pkl_target/best_target' + tag + '.pkl'))
                    
                pbar.set_postfix({'loss': online_loss})
                pbar.update()

        online_model.load_state_dict(torch.load('pkl/pkl_online/best_online' + tag + '.pkl'))
        online_model.eval()
        or_embeds, pr_embeds = online_model.embed(data.x,data.edge_index,slsp_adj,num_hop)
        embeds = or_embeds + pr_embeds
        scores = fit_logistic_regression_preset_splits(embeds.detach().cpu().numpy(), data.y.cpu().numpy(),train_masks, val_masks, test_masks)
        m = np.mean(scores)
        n = np.var(scores)
        print(m,n)
        with open(log_dir, 'a') as f:
                f.write('sgrl_mean: '+str(m)[0:7]+' sgrl_std: '+ str(n))
                f.write('\n')
        


       
    

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser('SGRL')
    parser.add_argument('--dataset_name', type=str, default='Wiki', help='dataset_name')
    parser.add_argument('--data_dir', type=str, default='../../datasets/Wiki', help='data_dir')
    parser.add_argument('--log_dir', type=str, default='./log/log_wiki.txt', help='log')
    parser.add_argument('--e1_lr', type=float, default=0.001, help='online_learning_rate')
    parser.add_argument('--e2_lr', type=float, default=0.001, help='target_learning_rate')
    parser.add_argument('--momentum', type=float, default=0.99, help='EMA')
    parser.add_argument('--weight_decay', type=float, default=0., help='weight_decay')
    parser.add_argument('--num_epochs', type=int, default=700, help='num_epochs')
    parser.add_argument('--seed', type=int, default=66666, help='seed')
    parser.add_argument('--hidden_dim', type=int, default=1024, help='hidden_dim')
    parser.add_argument('--num_layers', type=int, default=1, help='num_layers')
    parser.add_argument('--num_hop', type=int, default=1, help='num_hop')
    parser.add_argument('--trials', type=int, default=20, help='trials')
    args = parser.parse_args()  
    run(args)






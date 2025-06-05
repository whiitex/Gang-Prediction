from itertools import product
import time
import argparse
from datasets import get_dataset
import torch.nn.functional as F
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn import linear_model
from sklearn.neural_network import MLPClassifier
import warnings
warnings.filterwarnings("ignore")

from coarsening_opt2 import MultiLayerCoarsening
import torch
import os
from torch_geometric.data import DataLoader, DenseDataLoader as DenseLoader
from torch.optim import Adam
from train_eval_opt import cross_validation_with_val_set_opt, getMiddleRes, MLP

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--lr_decay_factor', type=float, default=0.5)
parser.add_argument('--lr_decay_step_size', type=int, default=50)

parser.add_argument('--opt_iters', type=int, default=10)
parser.add_argument('--eps', type=float, default=1.0)
parser.add_argument('--ratio', type=float, default=0.5)
parser.add_argument('--train', action='store_true')
parser.add_argument('--no_extra_mlp', action='store_true')


args = parser.parse_args()



layers = [1,2]
# hiddens = [16, 32, 64, 128]
hiddens = [64]
datasets = ['PROTEINS', 'MUTAG', 'NCI1', 'NCI109', 'IMDB-BINARY', 'IMDB-MULTI', 'DD']
nets = [MultiLayerCoarsening]

def train(model, optimizer, loader):
    model.train()

    total_loss = 0
    for data in loader:
        optimizer.zero_grad()
        data = data.to(device)
        # xs, new_adj, S, opt_loss = model(data, epsilon=0.01, opt_epochs=100)
        xs, new_adjs, Ss, opt_loss = model(data, epsilon=args.eps, opt_epochs=args.opt_iters)
        if opt_loss ==0.0:
            continue
        opt_loss.backward()
        total_loss += opt_loss.item() * num_graphs(data)
        optimizer.step()
    return total_loss / len(loader.dataset)

def eval_loss(model, loader):
    model.eval()

    loss = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            xs, new_adjs, Ss, opt_loss = model(data, epsilon=args.eps, opt_epochs=args.opt_iters)
        if opt_loss==0.0:
            continue
        loss += opt_loss.item()* num_graphs(data)
    return loss / len(loader.dataset), data.x, new_adjs, Ss


def eval_acc(model, dataset):
    model.eval()

    loader = DenseLoader(dataset, batch_size=args.batch_size, shuffle=False)
    Xs = []
    Ys = []
    for data in loader:
        data = data.to(device)
        Ys.append(data.y)
        with torch.no_grad():
            xs, new_adjs, Ss, opt_loss = model(data, epsilon=args.eps, opt_epochs=args.opt_iters)
            Xs.append(model.jump(xs))
            # Xs.append(xs[0])
    Xs = torch.cat(Xs, 0)
    Ys = torch.cat(Ys, 0)



    clf1 = linear_model.LogisticRegressionCV(solver='saga', multi_class='auto',max_iter=200,random_state=12345)
    cv = StratifiedKFold(10, random_state=12345)

    score1 = cross_val_score(clf1, X=Xs.detach().cpu().numpy(), y=Ys.detach().cpu().numpy(), cv=cv)

    clf2 = MLPClassifier(hidden_layer_sizes=(64,32), max_iter=400)
    score2 = cross_val_score(clf2, X=Xs.detach().cpu().numpy(), y=Ys.detach().cpu().numpy(), cv=cv)

    # print(score.mean(), score.std())
    return score1.mean(), score1.std(), score2.mean(), score2.std()


def num_graphs(data):
    if data.batch is not None:
        return data.num_graphs
    else:
        return data.x.size(0)



results = []
for dataset_name, Net in product(datasets, nets):
    best_result = (float('inf'), 0, 0)  # (loss, acc, std)
    print('-----\n{} - {}'.format(dataset_name, Net.__name__))
    for num_layers, hidden in product(layers, hiddens):
        dataset = get_dataset(dataset_name, sparse=False, dirname=None)
        model = Net(dataset, hidden, ratio=args.ratio)

        dirpath = "../savedmodels_eps"+str(args.eps)+"_iter"+str(args.opt_iters)
        if not os.path.exists(dirpath):
            os.mkdir(dirpath)
        # model_path = dirpath + "/opt_"+dataset_name+"_layers"+str(num_layers)+"_hidden"+str(hidden)+"_params.pkl"
        model_path = dirpath + "/opt_" + dataset_name + "_params.pkl"


        if args.train:
            #unsupervised training
            perm = torch.randperm(len(dataset))
            train_id = int(0.8*len(dataset))
            train_index = perm[:train_id]
            val_index = perm[train_id:]
            print("num_layers, hidden", num_layers, hidden)

            train_loader = DenseLoader(dataset[train_index], batch_size=args.batch_size, shuffle=True)
            model.to(device).reset_parameters()
            optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=0.0001) # adding a negative weight regularizaiton such that it cannot be zero.

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            t_start = time.perf_counter()
            val_losses = []
            val_loader = DenseLoader(dataset[val_index], batch_size=args.batch_size, shuffle=False)
            best_val_loss = 100000.0
            best_val_epoch = 0
            for epoch in range(1, args.epochs + 1):
                train_loss = train(model, optimizer, train_loader)
                val_loss, val_x, val_adjs, val_Ss = eval_loss(model, val_loader)

                val_losses.append(val_loss)
                eval_info = {
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'val_loss': val_losses[-1],
                }
                print(eval_info)
                if val_loss < best_val_loss:
                    torch.save(model.state_dict(), model_path)
                    best_val_loss = val_loss
                    best_val_epoch = epoch
                if epoch-best_val_epoch>30:
                    break
        #load model and test
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
            if args.no_extra_mlp:
                acc_mean1, acc_std1, acc_mean2, acc_std2 = eval_acc(model, dataset)
                print("Final Results LR: ", acc_mean1, acc_std1)
                print("Final Results MLP: ", acc_mean2, acc_std2)
            else:
                myData = getMiddleRes(dataset, model,args.batch_size, args.eps, args.opt_iters)
                mlp = MLP(myData.X.size(-1), 64, myData.num_classes)
                loss_mean, acc_mean, acc_std = cross_validation_with_val_set_opt(myData,mlp, 10, 200, 32, lr=0.01, weight_decay=0.0001)
        else:
            print(model_path)

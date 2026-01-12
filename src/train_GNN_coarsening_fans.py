import os
import sys
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "src")))

# torch
import torch
import torch.optim as optim
from torch_geometric.utils import to_networkx
import networkx as nx

# graph coarsening
from coarsening_utils_cost_fans import coarse_one_level_fans, calc_B
from graph_utils import *

# utils
from utils.utils import *
from utils.visualization import *

from GNN_model import GCN, train_gnn_1_epoch, evaluate_model
from coarsening_aware_loss import *


def train_GNN_coarsening_aware_loss(
    data: Data,
    levels: int,
    epoch_per_level: int,
    lr=0.01,
    wd=5e-4,
    method="variation_neighborhoods",
    algorithm: str = "greedy",
    K=50,
    nhid=128,
    dropout=0.1,
    device="cpu",
    max_cost_loss=1.0,
    candidates=None
):
    # model
    device = data.x.device
    N = data.num_nodes  # 2708
    nclass = len(torch.unique(data.y))  # 7
    model = GCN(nfeat=data.num_features, nhid=nhid, nclass=nclass, dropout=dropout).to(
        device
    )

    original_data = data
    Gc = data

    train_idx, val_idx, test_idx = create_train_val_test_split(original_data.num_nodes)
    P_train = create_P(train_idx, N, device=device)
    P_val = create_P(val_idx, N, device=device)
    P_test = create_P(test_idx, N, device=device)
    # criterion

    weight = Gc.y.bincount() / len(Gc.y)
    criterion = CoarseningAwareLoss(weight=weight)

    # train data
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    x, ycrs, yfine, ylosst, ylossv, valacc, num_nodes_coarse = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )

    # Create one-hot encoding of labels
    device = Gc.y.device
    num_classes = len(torch.unique(Gc.y))

    labels_onehot = torch.zeros(len(Gc.y), num_classes, device=device)
    labels_onehot.scatter_(1, Gc.y.view(-1, 1), 1)
    Gc.soft_y = labels_onehot

    ################################
    C = sparse_eye(N, device)
    Gc.W, Gc.L, Gc.dw = graph_params(Gc)
    # GG = to_networkx(Gc, to_undirected=True)
    # pos = nx.spring_layout(GG)
    # pos = [vals for vals in pos.values()]
    # Gc.pos = torch.tensor(pos, dtype=torch.float32, device=device)

    # colors and position by eigenvectors
    # L_dense = Gc.L.to_dense().cpu().numpy()
    # _, V = np.linalg.eigh(L_dense)
    # colors = V[:, :3]
    # colors = (colors - np.mean(colors, axis=0)) / (np.std(colors, axis=0) + 1e-8)
    # colors = (colors - colors.min(axis=0)) / (colors.max(axis=0) - colors.min(axis=0) + 1e-8)
    # Gc.colors = torch.tensor(colors, device=device)

    # positions (Laplacian eigenvectors)
    # pos = V[:, :2]
    # pos = (pos - np.mean(pos, axis=0)) / (np.std(pos, axis=0) + 1e-8)
    # pos = (pos - pos.min(axis=0)) / (pos.max(axis=0) - pos.min(axis=0) + 1e-8)
    # Gc.pos = torch.tensor(pos, device=device)

    B = calc_B(Gc, K)
    iC = None

    Call, Gall, iCs = [], [], []
    Gall.append(Gc)
    # Call.append(C)

    bar = tqdm(range(1, levels+1), desc="Coarsening Levels")
    for level in bar:

        # get embeddings from the GNN
        embeddings = model.get_embeddings(Gc.x, Gc.W)
        Gc.embeddings = F.normalize(embeddings, p=2, dim=1)

        ratio = np.log(level ** (4 / 3)) / 100 + 0.015

        iC, Gc, B = coarse_one_level_fans(
            Gc,
            iC,
            B,
            K=K,
            method=method,
            algorithm=algorithm,
            level=level,
            r_cur=ratio,
            candidates=candidates,
        )
        
        C = torch.sparse.mm(iC, C)
        Gall.append(Gc)
        Call.append(C)
        iCs.append(iC)

        # update candidates
        coarse_mat = iC.coalesce()
        row, col = coarse_mat.indices()
        supernode = {}
        for r, c in zip(row.tolist(), col.tolist()):
            supernode[c] = r
        
        for i in range(len(candidates)):
            candidates[i] = [supernode[x] for x in candidates[i]]
            candidates[i] = list(set(candidates[i]))
        
        candidates = [candidate for candidate in candidates if len(candidate) > 1]

        # traing GNN
        for epoch in range(epoch_per_level):
            # train the GNN on the coarsened graph
            # C_train = C @ P_train
            CC = C * C
            C_train = CC @ P_train
            mask = torch.sum(C_train, dim=1).to_dense() > 1e-5
            train_idx_c = torch.nonzero(mask).view(-1)
            test_idx_c = torch.nonzero(~mask).view(-1)
            y_soft_train = C_train @ original_data.soft_y[train_idx]
            Gc.y_train = torch.argmax(y_soft_train, dim=1)
            Gc.y = torch.argmax(Gc.soft_y, dim=1) if Gc.soft_y is not None else Gc.y

            train_loss, train_acc = train_gnn_1_epoch(
                model,
                optimizer,
                criterion,
                Gc,
                train_idx_c,
                coarse_loss=epoch == 0,
            )

        # evaluate the model on the coarsened graph
        acc_test_c, pred_test_c, pred = evaluate_model(
            model, Gc, test_idx_c, log_info=False
        )
        acc_test, pred_test, _ = evaluate_model(
            model, original_data, test_idx, log_info=False
        )


        C_plus = torch.sparse_coo_tensor(
            torch.flip(CC.indices(), dims=[0]),
            torch.ones_like(CC.values()),
            (CC.size(1), CC.size(0)),
        )
        pred_fine = torch.argmax(C_plus @ pred, dim=1)

        accuracy_fine = torch.sum(
            pred_fine[test_idx] == original_data.y[test_idx]
        ).item() / len(test_idx)


        # ploting data
        x.append(epoch + 1)
        ycrs.append(acc_test_c)
        yfine.append(accuracy_fine)
        ylosst.append(train_loss)
        # ylossv.append(validation_loss)
        # valacc.append(validation_accuracy)
        num_nodes_coarse.append(Gc.num_nodes)

        # count gangs classified correctly on coarse
        print(f"{pred_test_c}, tot: {len(pred_test_c)}, 1: {sum(pred_test_c.tolist())}")
        # gangs = Gc.y[test_idx_c].sum().item()
        # pred_gangs = [1 if i > .5 else -1 for i in pred_test_c.tolist()]
        # correct_gangs = sum([pred_gangs == Gc.y[test_idx_c].tolist()])

        bar.set_postfix_str(
            f"{epoch_per_level}, r: {ratio:.2f}, nodes: {Gc.num_nodes} "
            # + f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}, "
            + f"coarse: {acc_test_c:.4f}, orig: {acc_test:.4f} fine: {accuracy_fine:.4f}"
            # + f" gangs: {correct_gangs}/{gangs} "
        )


    name = f"data_gnn_CoarseningAwareLoss_V2_epochs_{epoch_per_level}.npy"
    np.save(
        f"{save_path}{name}",
        {
            "x": x,
            "ycrs": ycrs,
            "yfine": yfine,
            "ylosst": ylosst,
            "Gall": Gall,
            "Call": Call,
            "ylossv": ylossv,
            "valacc": valacc,
            "num_nodes_coarse": num_nodes_coarse,
            "description": f"Data obtained using: {method=}, CoarseningAwareLoss() levels approach",
        },
    )

    return Gall, Call, iCs

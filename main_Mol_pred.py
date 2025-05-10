import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score, mean_absolute_error, mean_squared_error

import torch
import torch.nn as nn
import torch.optim as optim

from Molprop_dataset.MoleculeNet_Graph import MoleculeNetGraphDataset, CustomDataset
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import RandomSampler
from functools import partial
from Molprop_dataset.collator_prop import collator, Batch

import torch.multiprocessing

from Molprop_dataset.splitters import scaffold_split
from Molprop_dataset.utils import get_num_task_and_type, get_molecule_repr_MoleculeSTM
from Molprop_dataset.molecule_graph_model import Graph_pred

# 设置共享策略
torch.multiprocessing.set_sharing_strategy('file_system')


def train_classification(model, device, loader, optimizer):
    if args.training_mode == "fine_tuning":
        model.train()
    else:
        model.eval()
    linear_model.train()
    total_loss = 0

    if args.verbose:
        L = tqdm(loader)
    else:
        L = loader
    for step, batch in enumerate(L):

        molecule_repr, _ = model(batch)
        pred = linear_model(molecule_repr)
        pred = pred.float()

        y = batch.y
        y = torch.stack(y)
        y = y.view(pred.shape).to(device).float()

        is_valid = y ** 2 > 0
        loss_mat = criterion(pred, (y + 1) / 2)
        loss_mat = torch.where(
            is_valid, loss_mat,
            torch.zeros(loss_mat.shape).to(device).to(loss_mat.dtype))

        optimizer.zero_grad()
        loss = torch.sum(loss_mat) / torch.sum(is_valid)
        loss.backward()
        optimizer.step()
        total_loss += loss.detach().item()

    return total_loss / len(loader)


@torch.no_grad()
def eval_classification(model, device, loader):
    model.eval()
    linear_model.eval()
    y_true, y_scores = [], []

    if args.verbose:
        L = tqdm(loader)
    else:
        L = loader
    for step, batch in enumerate(L):

        molecule_repr, _ = model(batch)
        pred = linear_model(molecule_repr)
        pred = pred.float()
        y = batch.y
        y = torch.stack(y)
        y = y.view(pred.shape).to(device).float()

        y_true.append(y)
        y_scores.append(pred)

    y_true = torch.cat(y_true, dim=0).cpu().numpy()
    y_scores = torch.cat(y_scores, dim=0).cpu().numpy()

    roc_list = []
    for i in range(y_true.shape[1]):
        # AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == -1) > 0:
            is_valid = y_true[:, i] ** 2 > 0
            roc_list.append(roc_auc_score((y_true[is_valid, i] + 1) / 2, y_scores[is_valid, i]))
        else:
            print("{} is invalid".format(i))

    if len(roc_list) < y_true.shape[1]:
        print(len(roc_list))
        print("Some target is missing!")
        print("Missing ratio: %f" %(1 - float(len(roc_list)) / y_true.shape[1]))

    return sum(roc_list) / len(roc_list), 0, y_true, y_scores


def train_regression(model, device, loader, optimizer):
    if args.training_mode == "fine_tuning":
        model.train()
    else:
        model.eval()
    linear_model.train()
    total_loss = 0

    if args.verbose:
        L = tqdm(loader)
    else:
        L = loader
    for step, batch in enumerate(L):

        molecule_repr, _ = model(batch)
        pred = linear_model(molecule_repr)
        pred = pred.float()
        y = batch.y
        y = torch.stack(y)
        y = y.view(pred.shape).to(device).float()

        loss = criterion(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.detach().item()

    return total_loss / len(loader)


@torch.no_grad()
def eval_regression(model, device, loader):
    model.eval()
    y_true, y_pred = [], []

    if args.verbose:
        L = tqdm(loader)
    else:
        L = loader
    for step, batch in enumerate(L):

        molecule_repr, _ = model(batch)
        pred = linear_model(molecule_repr)
        pred = pred.float()
        y = batch.y
        y = torch.stack(y)
        y = y.view(pred.shape).to(device).float()

        y_true.append(y)
        y_pred.append(pred)

    y_true = torch.cat(y_true, dim=0).cpu().numpy()
    y_pred = torch.cat(y_pred, dim=0).cpu().numpy()
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mae = mean_absolute_error(y_true, y_pred)
    return {'RMSE': rmse, 'MAE': mae}, y_true, y_pred


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=int, default=1)
    parser.add_argument("--training_mode", type=str, default="fine_tuning", choices=["fine_tuning", "linear_probing"])
    parser.add_argument("--molecule_type", type=str, default="Graph", choices=["SMILES", "Graph"])

    ########## for dataset and split ##########
    parser.add_argument("--dataspace_path", type=str, default="data")
    # parser.add_argument("--dataset", type=str, default="bace")
    parser.add_argument("--dataset", type=str, default="hiv")
    parser.add_argument("--split", type=str, default="scaffold")

    ########## for optimization ##########
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--lr_scale", type=float, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)  # 100
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--schedule", type=str, default="cycle")
    parser.add_argument("--warm_up_steps", type=int, default=10)


    ########## for Graphormer ##########
    parser.add_argument('--temperature', type=float, default=0.1, help='the temperature of NT_XentLoss')
    parser.add_argument('--graph_hidden_dim', type=int, default=768, help='')
    parser.add_argument('--bert_hidden_dim', type=int, default=768, help='')
    parser.add_argument('--drop_ratio', default=0.1, type=float)
    parser.add_argument('--projection_dim', type=int, default=256)

    ########## for saver ##########
    parser.add_argument("--eval_train", type=int, default=0)
    parser.add_argument("--verbose", type=int, default=0)

    parser.add_argument("--input_model_path", type=str, default=None)
    parser.add_argument("--output_model_dir", type=str, default='save_model/task2_prop/hiv')

    args = parser.parse_args()
    print("arguments\t", args)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda:" + str(args.device))
    print(device)

    num_tasks, task_mode = get_num_task_and_type(args.dataset)
    dataset_folder = os.path.join(args.dataspace_path, "MoleculeNet_data", args.dataset)

    data_processed_path = os.path.join(args.dataspace_path, "MoleculeNet_data", args.dataset, "processed",
                                       "data_processed.pt")
    dataset = CustomDataset(data_processed_path)
    use_pyg_dataset = False

    assert args.split == "scaffold"
    print("split via scaffold")
    smiles_list = pd.read_csv(
        dataset_folder + "/processed/smiles.csv", header=None)[0].tolist()
    train_dataset, valid_dataset, test_dataset = scaffold_split(
        dataset, smiles_list, null_value=0, frac_train=0.8,
        frac_valid=0.1, frac_test=0.1, pyg_dataset=use_pyg_dataset)

    train_sampler = RandomSampler(train_dataset)
    train_loader = DataLoader(train_dataset, sampler=train_sampler,
                                  batch_size=args.batch_size,
                                  num_workers=0, pin_memory=False, drop_last=True,
                                  collate_fn=partial(collator, max_node=512,
                                                     multi_hop_max_dist=20, spatial_pos_max=20))
    val_loader = DataLoader(valid_dataset, shuffle=False,
                                batch_size=args.batch_size,
                                num_workers=0, pin_memory=False, drop_last=True,
                                collate_fn=partial(collator, max_node=512,
                                                   multi_hop_max_dist=20, spatial_pos_max=20))
    test_loader = DataLoader(test_dataset, shuffle=False,
                                 batch_size=args.batch_size,
                                 num_workers=0, pin_memory=False, drop_last=True,
                                 collate_fn=partial(collator, max_node=512,
                                                    multi_hop_max_dist=20, spatial_pos_max=20))


    ckpt_path = 'save_model/prompt_gp_v4_pretrain_200k/'
    model_ckpt = torch.load(ckpt_path)
    model = Graph_pred(
        temperature=args.temperature,
        drop_ratio=args.drop_ratio,
        graph_hidden_dim=args.graph_hidden_dim,
        bert_hidden_dim=args.bert_hidden_dim,
        # bert_pretrain,
        projection_dim=args.projection_dim,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_tasks=num_tasks
    )
    model.load_state_dict(model_ckpt, False)
    molecule_dim = args.graph_hidden_dim
    model.to(device)


    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    linear_model = nn.Linear(molecule_dim, num_tasks).to(device)

    # set up optimizer
    if args.training_mode == "fine_tuning":
        model_param_group = [
            {"params": model.parameters()},
            {"params": linear_model.parameters(), 'lr': args.lr * args.lr_scale}
        ]
    else:
        model_param_group = [
            {"params": linear_model.parameters(), 'lr': args.lr * args.lr_scale}
        ]
    optimizer = optim.Adam(model_param_group, lr=args.lr, weight_decay=args.weight_decay)

    if task_mode == "classification":
        train_func = train_classification
        eval_func = eval_classification

        train_roc_list, val_roc_list, test_roc_list = [], [], []
        train_acc_list, val_acc_list, test_acc_list = [], [], []
        best_val_roc, best_val_idx = -1, 0
        criterion = nn.BCEWithLogitsLoss(reduction="none")

        for epoch in range(1, args.epochs + 1):
            loss_acc = train_func(model, device, train_loader, optimizer)
            print("Epoch: {}\nLoss: {}".format(epoch, loss_acc))

            if args.eval_train:
                train_roc, train_acc, train_target, train_pred = eval_func(model, device, train_loader)
            else:
                train_roc = train_acc = 0
            val_roc, val_acc, val_target, val_pred = eval_func(model, device, val_loader)
            test_roc, test_acc, test_target, test_pred = eval_func(model, device, test_loader)

            train_roc_list.append(train_roc)
            train_acc_list.append(train_acc)
            val_roc_list.append(val_roc)
            val_acc_list.append(val_acc)
            test_roc_list.append(test_roc)
            test_acc_list.append(test_acc)
            print("train: {:.6f}\tval: {:.6f}\ttest: {:.6f}".format(train_roc, val_roc, test_roc))
            print()

            if val_roc > best_val_roc:
                best_val_roc = val_roc
                best_val_idx = epoch - 1
                if args.output_model_dir is not None:
                    ##### save best model #####
                    output_model_path = os.path.join(args.output_model_dir, "{}_model.pth".format(args.dataset))
                    saved_model_dict = {
                        "model": model.state_dict()
                    }
                    torch.save(saved_model_dict, output_model_path)

                    filename = os.path.join(args.output_model_dir, "{}_evaluation.pth".format(args.dataset))
                    np.savez(
                        filename, val_target=val_target, val_pred=val_pred,
                        test_target=test_target, test_pred=test_pred)

        print("best train: {:.6f}\tval: {:.6f}\ttest: {:.6f}".format(train_roc_list[best_val_idx], val_roc_list[best_val_idx], test_roc_list[best_val_idx]))

    else:
        train_func = train_regression
        eval_func = eval_regression
        criterion = torch.nn.MSELoss()
        
        train_result_list, val_result_list, test_result_list = [], [], []
        metric_list = ['RMSE', 'MAE']
        best_val_rmse, best_val_idx = 1e10, 0

        for epoch in range(1, args.epochs + 1):
            loss_acc = train_func(model, device, train_loader, optimizer)
            print('Epoch: {}\nLoss: {}'.format(epoch, loss_acc))

            if args.eval_train:
                train_result, train_target, train_pred = eval_func(model, device, train_loader)
            else:
                train_result = {'RMSE': 0, 'MAE': 0, 'R2': 0}
            val_result, val_target, val_pred = eval_func(model, device, val_loader)
            test_result, test_target, test_pred = eval_func(model, device, test_loader)

            train_result_list.append(train_result)
            val_result_list.append(val_result)
            test_result_list.append(test_result)

            for metric in metric_list:
                print('{} train: {:.6f}\tval: {:.6f}\ttest: {:.6f}'.format(metric, train_result[metric], val_result[metric], test_result[metric]))
            print()

            if val_result['RMSE'] < best_val_rmse:
                best_val_rmse = val_result['RMSE']
                best_val_idx = epoch - 1
                if args.output_model_dir is not None:
                    ##### save best model #####
                    output_model_path = os.path.join(args.output_model_dir, "{}_model_best.pth".format(args.dataset))
                    saved_model_dict = {
                        'model': model.state_dict()
                    }
                    torch.save(saved_model_dict, output_model_path)

                    filename = os.path.join(args.output_model_dir, "{}_evaluation_best.pth".format(args.dataset))
                    np.savez(
                        filename, val_target=val_target, val_pred=val_pred,
                        test_target=test_target, test_pred=test_pred)

        for metric in metric_list:
            print('Best (RMSE), {} train: {:.6f}\tval: {:.6f}\ttest: {:.6f}'.format(
                metric, train_result_list[best_val_idx][metric], val_result_list[best_val_idx][metric], test_result_list[best_val_idx][metric]))


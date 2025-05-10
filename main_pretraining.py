import argparse
from config import parse_args

import random
import torch
import numpy as np
from torch.autograd import Variable
from tqdm import tqdm
import sys

from data_provider.GraST_dataset_0313 import GraSTPretrain_STM

from torch.utils.data import Dataset, DataLoader
from data_provider.collator_0313 import collator, Batch
import torch_geometric
# from model.contrastive_GraST_0313 import GraSTSimclr
from model.contrastive_GraST_gp import GraSTSimclr
from optimization import BertAdam, warmup_linear
from torch.utils.data import RandomSampler
import os
import re
import statistics

# LiuC:
from functools import partial
# LiuC:
import logging
import time



def prepare_model_and_optimizer(args, device):

    model = GraSTSimclr(
        temperature=args.temperature,
        drop_ratio=args.drop_ratio,
        graph_hidden_dim=args.graph_hidden_dim,
        bert_hidden_dim=args.bert_hidden_dim,
        # bert_pretrain,
        projection_dim=args.projection_dim,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )


    model.to(device)
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [
                p for n, p in param_optimizer
                if not any(nd in n for nd in no_decay)
            ],
            'weight_decay': 0.01
        },
        {
            'params': [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            'weight_decay': 0.0
        },
    ]

    optimizer = BertAdam(
        optimizer_grouped_parameters,
        weight_decay=args.weight_decay,
        lr=args.lr,
        warmup=args.warmup,
        t_total=args.total_steps,
    )

    return model, optimizer

def Eval(model, dataloader, device, args):
    model.eval()
    with torch.no_grad():
        acc1 = 0
        acc2 = 0
        allcnt = 0
        graph_rep_total = None
        text_rep_total = None
        for idx, batch in enumerate((dataloader)):

            text_rep, graph_rep, loss = model.training_step(batch_idx=idx, batch=batch)

            scores1 = torch.cosine_similarity(
                graph_rep.unsqueeze(1).expand(graph_rep.shape[0], graph_rep.shape[0], graph_rep.shape[1]),
                text_rep.unsqueeze(0).expand(text_rep.shape[0], text_rep.shape[0], text_rep.shape[1]), dim=-1)
            scores2 = torch.cosine_similarity(
                text_rep.unsqueeze(1).expand(text_rep.shape[0], text_rep.shape[0], text_rep.shape[1]),
                graph_rep.unsqueeze(0).expand(graph_rep.shape[0], graph_rep.shape[0], graph_rep.shape[1]), dim=-1)

            argm1 = torch.argmax(scores1, axis=1)
            argm2 = torch.argmax(scores2, axis=1)

            acc1 += sum((argm1 == torch.arange(argm1.shape[0]).to(device)).int()).item()
            acc2 += sum((argm2 == torch.arange(argm2.shape[0]).to(device)).int()).item()

            allcnt += argm1.shape[0]

            if graph_rep_total is None or text_rep_total is None:
                graph_rep_total = graph_rep
                text_rep_total = text_rep
            else:
                graph_rep_total = torch.cat((graph_rep_total, graph_rep), axis=0)
                text_rep_total = torch.cat((text_rep_total, text_rep), axis=0)

    np.save(f'{args.output_path}/graph_rep.npy', graph_rep_total.cpu())
    np.save(f'{args.output_path}/text_rep.npy', text_rep_total.cpu())

    return acc1 / allcnt, acc2 / allcnt


# get every sentence's rep
def CalSent(model, dataloader, device, args):
    model.eval()
    with torch.no_grad():
        text_rep_total = None
        for batch in (dataloader):
            text, mask = batch
            text = text.to(device)
            mask = mask.to(device)
            text_rep = model.text_encoder(text, mask)
            text_rep = model.text_proj_head(text_rep)

            if text_rep_total is None:
                text_rep_total = text_rep
            else:
                text_rep_total = torch.cat((text_rep_total, text_rep), axis=0)

    np.save(f'{args.output_path}/text_rep.npy', text_rep_total.cpu())



def setup_logging():
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO)
    logger = logging.getLogger(__name__)

    return logger

def main():
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    device = torch.device(f'cuda:{args.device}')
    model, optimizer = prepare_model_and_optimizer(args, device)

    # LiuC: 通过text文件名中的序号 获取ids
    ids = []
    text_name_list = os.listdir("preprocessing/PubChem/PubChem_process/text_PubChem")
    for text_name in text_name_list:
        text_id = re.split('[_.]', text_name)[1]
        text_id = int(text_id)
        ids.append(text_id)
    ids.sort()
    seq = np.arange(len(ids))
    print(seq)
    print(len(seq))
    np.random.shuffle(seq)
    print(seq)
    print(len(seq))

    scaf = []
    k = int(len(seq) / 10)

    scaf.append(seq[:9 * k])
    scaf.append(seq[9 * k:])

    TrainSet = GraSTPretrain_STM(args, ids, scaf[0])
    DevSet = GraSTPretrain_STM(args, ids, scaf[1])

    train_sampler = RandomSampler(TrainSet)

    train_dataloader = DataLoader(TrainSet, sampler=train_sampler,
                                  batch_size=args.batch_size,
                                  num_workers=4, pin_memory=True, drop_last=True,
                                  collate_fn=partial(collator, max_node=128,
                                                     multi_hop_max_dist=20, spatial_pos_max=20))
    dev_dataloader = DataLoader(DevSet, shuffle=False,
                                batch_size=args.batch_size,
                                num_workers=4, pin_memory=True, drop_last=True,
                                collate_fn=partial(collator, max_node=128,
                                                   multi_hop_max_dist=20, spatial_pos_max=20))
    global_step = 0
    tag = True
    best_acc = 0
    # start_time = time.time()
    num_total_steps = len(train_dataloader) * args.epoch

    print(args.epoch)
    for epoch in range(args.epoch):
        if tag == False:
            break
        print('Epoch:', epoch)
        acc1, acc2 = Eval(model, dev_dataloader, device, args)
        print('Epoch:', epoch, ', DevAcc1:', acc1)
        print('Epoch:', epoch, ', DevAcc2:', acc2)
        if acc1 > best_acc:
            best_acc = acc1
            torch.save(model.state_dict(), f'{args.output_path}/model_{epoch}_{seed}_{acc1}_{acc2}.ckpt')
            print('Save checkpoint ', global_step)
        acc = 0
        allcnt = 0
        sumloss = 0
        model.train()
        for idx, batch in enumerate((train_dataloader)):
            text_rep, graph_rep, loss = model.training_step(batch_idx=idx, batch=batch)
            print('Epoch:', epoch, ', global_step:', global_step, ', loss:', loss)
            scores = text_rep.mm(graph_rep.t())
            argm = torch.argmax(scores, axis=1)
            acc += sum((argm == torch.arange(argm.shape[0]).to(device)).int()).item()
            allcnt += argm.shape[0]
            sumloss += loss.item()
            loss.backward()
            # if idx%4==1:
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

            if global_step > args.total_steps:
                tag = False
                break
        optimizer.step()
        optimizer.zero_grad()
        print('Epoch:', epoch, ', Acc:', acc / allcnt, ', Loss:', sumloss / allcnt)


if __name__ == "__main__":
    args = parse_args()
    print(args)
    setup_logging()

    for seed in [42]:
        args.seed = seed
        print(f'seed:{args.seed}')
        main()


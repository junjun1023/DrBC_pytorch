import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import time

def top_n_percentage(score_gt, score_pr, k, device):
    
        if not isinstance(score_gt, torch.Tensor):
                score_gt = torch.from_numpy(score_gt)
        score_gt = score_gt.to(device)
        score_gt = torch.reshape(score_gt, (-1, ))
        
        if not isinstance(score_pr, torch.Tensor):
                score_pr = torch.from_numpy(score_pr)
        score_pr = score_pr.to(device)
        score_pr = torch.reshape(score_pr, (-1, ))
        
        nodes = score_gt.size()[0]
        k = int(nodes * k / 100)
        
        gt_value, gt_indice = torch.topk(score_gt, k)
        pr_value, pr_indice = torch.topk(score_pr, k)

        gt_indice = set(gt_indice.cpu().numpy())
        pr_indice = set(pr_indice.cpu().numpy())

        intersect = len(gt_indice & pr_indice)
        top = intersect/k
        
        return top




def kendal_tau_distance(score_gt, score_pr):
        from scipy import stats
        if isinstance(score_gt, torch.Tensor):
                score_gt = torch.reshape(score_gt, (-1, ))
                score_gt = score_gt.cpu().detach().numpy()
        elif isinstance(score_gt, list):
                score_gt = np.array(score_gt)
                
        if isinstance(score_pr, torch.Tensor):
                score_pr = torch.reshape(score_pr, (-1, ))
                score_pr = score_pr.cpu().detach().numpy()
        elif isinstance(score_pr, list):
                score_pr = np.array(score_pr)
        
        tau, p_value = stats.kendalltau(score_gt, score_pr)
        return tau


def eval_model(model, dataloader, device):
        
        model = model.eval().to(device)
        
        top1_list = []
        top5_list = []
        top10_list = []
        kendal_list = []
        loss_list = []
        time_list = []

        for batch in tqdm(dataloader):

                batch = batch.to(device)
                
                start = time.time()
                score_pr = model(batch)
                end = time.time()

                b_index = batch.batch.cpu().numpy()
                b = np.max(b_index) + 1

                for b_ in range(b):

                        indice, = np.where(b_index == b_)


                        gt = batch.y[indice].squeeze()
                        pr = score_pr[indice].squeeze()

                        # evaluation
                        top1 = top_n_percentage(gt, pr, k=1, device=device)
                        top5 = top_n_percentage(gt, pr, k=5, device=device)
                        top10 = top_n_percentage(gt, pr, k=10, device=device)
                        kendal = kendal_tau_distance(gt, pr)

                        # compute loss
                        src = np.random.choice(len(indice), 5*len(indice), replace=True)
                        det = np.random.choice(len(indice), 5*len(indice), replace=True)
                        src = torch.from_numpy(src)
                        det = torch.from_numpy(det)

                        y_gt = gt[det] - gt[src]
                        y_pr = pr[det] - pr[src]

                        y_gt = nn.Sigmoid()(y_gt)
                        y_pr = nn.Sigmoid()(y_pr)

                        loss = nn.BCELoss()(y_pr, y_gt)

                top1_list.append(top1)
                top5_list.append(top5)
                top10_list.append(top10)
                kendal_list.append(kendal)
                loss_list.append(loss.item())
                time_list.append(end-start)
        
        
        return top1_list, top5_list, top10_list, kendal_list, time_list, loss_list

import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
l2 = False
rho = 5e-2
alpha = 5e-4
def regularized_nll_loss(args, model, output, target):
    index = 0
    loss = F.nll_loss(output, target)
    if args.l2:
        for name, param in model.named_parameters():
            if name.split('.')[-1] == "weight" and name.split('.')[-2]=="conv":
                loss += args.alpha * param.norm()
                index += 1
    return loss

def admm_loss(device, model, Z, U, nll_loss):
    
    idx = 0
    loss = nll_loss
    admm = 0.0
    for name, param in model.named_parameters():
        if name.split('.')[-1][0:6] == "weight"and name.split('.')[-2]=="conv":
            u = U[idx].to(device)
            z = Z[idx].to(device)
            
            loss += rho / 2 * (param - z + u).norm()
            admm = rho / 2 * (param - z + u).norm()
            if l2:
                loss += alpha * param.norm()
            idx += 1
    return loss

def initialize_Z_and_U(model):
    Z = ()
    U = ()
    for name, param in model.named_parameters():
        if name.split('.')[-1] == "weight" and name.split('.')[-2]=="conv":
            Z += (param.detach().cpu().clone(),)
            U += (torch.zeros_like(param).cpu(),)
    return Z, U


def update_X(model):
    X = ()
    for name, param in model.named_parameters():
        if name.split('.')[-1] == "weight" and name.split('.')[-2]=="conv":
            X += (param.detach().cpu().clone(),)
    return X


def update_Z(X, U,percents):
    new_Z = ()
    idx = 0
    for x, u in zip(X, U):
        z = x + u
        pcen = np.percentile(abs(z), 100*percents)
        under_threshold = abs(z) < pcen
        z.data[under_threshold] = 0
        new_Z += (z,)
        idx += 1
    return new_Z


def update_Z_l1(X, U):
    new_Z = ()
    delta = alpha / rho
    for x, u in zip(X, U):
        z = x + u
        new_z = z.clone()
        if (z > delta).sum() != 0:
            new_z[z > delta] = z[z > delta] - delta
        if (z < -delta).sum() != 0:
            new_z[z < -delta] = z[z < -delta] + delta
        if (abs(z) <= delta).sum() != 0:
            new_z[abs(z) <= delta] = 0
        new_Z += (new_z,)
    return new_Z


def update_U(U, X, Z):
    new_U = ()
    for u, x, z in zip(U, X, Z):
        new_u = u + x - z
        new_U += (new_u,)
    return new_U


def prune_weight(weight, device, percent):
    # to work with admm, we calculate percentile based on all elements instead of nonzero elements.
    weight_numpy = weight.detach().cpu().numpy()
    pcen = np.percentile(abs(weight_numpy), 100*percent)
    under_threshold = abs(weight_numpy) < pcen
    weight_numpy[under_threshold] = 0
    mask = torch.Tensor(abs(weight_numpy) >= pcen).to(device)
    return mask


def prune_l1_weight(weight, device, delta):
    weight_numpy = weight.detach().cpu().numpy()
    under_threshold = abs(weight_numpy) < delta
    weight_numpy[under_threshold] = 0
    mask = torch.Tensor(abs(weight_numpy) >= delta).to(device)
    return mask

def apply_prune(param_groups, device,percents):
    # returns dictionary of non_zero_values' indices
    print("Apply Pruning based on percentile")
    dict_mask = {}
    for i,group in enumerate(param_groups):
        if(i==1):
            for j, p in enumerate(group['params']):
                mask = prune_weight(p, device, percents)
                dict_mask[j] = mask
                # print("mask",mask)
    return dict_mask
# def apply_prune(model, device):
#     # returns dictionary of non_zero_values' indices
#     print("Apply Pruning based on percentile")
#     dict_mask = {}
#     idx = 0
#     for name, param in model.named_parameters():
#         # print("name:",name)
#         # print("param",param)
#         if name.split('.')[-1] == "weight" and name.split('.')[-2]=="conv":
#             mask = prune_weight(param, device, percents)
#             param.data.mul_(mask)
#             # param.data = torch.Tensor(weight_pruned).to(device)
#             dict_mask[idx] = mask
#         else:
#             mask = prune_weight(param, device, 0)
#             dict_mask[idx] = mask
#         # print("mask:",dict_mask[idx])
#         # print("param.shape", param.shape)
#         # print("mask.shape", dict_mask[idx].shape)
#         idx += 1
#     return dict_mask


def apply_l1_prune(model, device):
    delta = alpha / rho
    print("Apply Pruning based on percentile")
    dict_mask = {}
    idx = 0
    for name, param in model.named_parameters():
        if name.split('.')[-1] == "weight" and name.split('.')[-2]=="conv":
            mask = prune_l1_weight(param, device, delta)
            param.data.mul_(mask)
            dict_mask[name] = mask
            idx += 1
    return dict_mask


def print_convergence(model, X, Z):
    idx = 0
    print("normalized norm of (weight - projection)")
    norm_sum = 0.0 
    for name, _ in model.named_parameters():
        if name.split('.')[-1] == "weight" and name.split('.')[-2]=="conv":
            x, z = X[idx], Z[idx]
            norm_sum = norm_sum + (x-z).norm().item() / x.norm().item()
            
            idx += 1
    print("norm_sum = ", norm_sum, "norm_avg = ", norm_sum/idx)
def print_prune(model):
    prune_param, total_param = 0, 0
    for name, param in model.named_parameters():
        if name.split('.')[-1] == "weight" and name.split('.')[-2]=="conv":
            print("[at weight {}]".format(name))
            print("percentage of pruned: {:.4f}%".format(100 * (abs(param) == 0).sum().item() / param.numel()))
            print("nonzero parameters after pruning: {} / {}\n".format((param != 0).sum().item(), param.numel()))
        total_param += param.numel()
        prune_param += (param != 0).sum().item()
    print("total nonzero parameters after pruning: {} / {} ({:.4f}%)".
          format(prune_param, total_param,
                 100 * (total_param - prune_param) / total_param))
# def print_prune(param_groups):
#     prune_param, total_param = 0, 0
#     for i,group in enumerate(param_groups):
#         if(i==1):
#             for j, param in enumerate(group['params']):
#                 print("percentage of pruned: {:.4f}%".format(100 * (abs(param) == 0).sum().item() / param.numel()))
#                 print("nonzero parameters after pruning: {} / {}\n".format((param != 0).sum().item(), param.numel()))
#             total_param += param.numel()
#             prune_param += (param != 0).sum().item()
#     print("total nonzero parameters after pruning: {} / {} ({:.4f}%)".
#           format(prune_param, total_param,
#                  100 * (total_param - prune_param) / total_param))

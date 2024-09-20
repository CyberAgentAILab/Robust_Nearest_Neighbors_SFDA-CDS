from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import random
import argparse
import numpy as np
import torchvision.transforms as transforms
import wandb
import pickle
import time
from sklearn.metrics import accuracy_score
from torch.optim.lr_scheduler import *
from utils import *
from os.path import join
from os import makedirs
from datasets import *
from model import *
from moco import *

from transformers import SwinModel

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--data_dir', type=str, default='data')
parser.add_argument('--dataset', default='visdac/target', type=str)
parser.add_argument('--source', default='visdac_source', type=str)
parser.add_argument('--weights', type=str)
parser.add_argument('--noisy_path', type=str, default=None)

parser.add_argument('--num_neighbors', default=100, type=int)
parser.add_argument('--num_near_neighbors', default=10, type=int)
parser.add_argument('--num_class', default=10, type=int)
parser.add_argument('--temporal_length', default=5, type=int)

parser.add_argument('--batch_size', default=256, type=int, help='train batchsize') 
parser.add_argument('--lr', '--learning_rate', default=0.02, type=float, help='initial learning rate')
parser.add_argument('--num_epochs', default=300, type=int)
parser.add_argument('--temperature', default=0.07, type=float, help='softmax temperature (default: 0.07)')

parser.add_argument('--seed', default=123)
parser.add_argument('--gpuid', default=0, type=int)

parser.add_argument('--ctr', action='store_false', help="use contrastive loss")
parser.add_argument('--label_refinement', action='store_false', help="Use label refinement")
parser.add_argument('--neg_l', action='store_false', help="Use negative learning")
parser.add_argument('--reweighting', action='store_false', help="Use reweighting")

parser.add_argument('--pretrained', default=None, type=str)

parser.add_argument('--run_name', type=str)
parser.add_argument('--wandb', action='store_true', help="Use wandb")

args = parser.parse_args()

torch.cuda.set_device(args.gpuid)
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

if args.wandb:
    wandb.init(project="Improving pseudolabel refinement for imbalanced source-free domain adaptation", name = args.run_name)

def entropy(p, axis=1):
    return -torch.sum(p * torch.log2(p+1e-5), dim=axis)

def get_distances(X, Y, dist_type="cosine"):
    if dist_type == "euclidean":
        distances = torch.cdist(X, Y)
    elif dist_type == "cosine":
        distances = 1 - torch.matmul(F.normalize(X, dim=1), F.normalize(Y, dim=1).T)
    else:
        raise NotImplementedError(f"{dist_type} distance not implemented.")

    return distances

def histogram(labels):
    hist = torch.bincount(labels)
    if len(hist) < args.num_class:
        hist = F.pad(hist, (0, args.num_class-len(hist)), "constant", 0)
        
    return hist

@torch.no_grad()
def soft_k_nearest_neighbors(features, features_gen, features_bank, features_gen_bank, probs_bank, labels, labels_bank):
    pred_probs = []
    pred_probs_all = []
    avg_nn = 0
    
    for feats, feats_gen in zip(features.split(64), features_gen.split(64)):
        distances = get_distances(feats, features_bank)
        _, idxs_src = distances.sort()
        idxs = idxs_src[:, : args.num_near_neighbors]
        idxs_src = idxs_src[:, : args.num_neighbors]
        distances_gen = get_distances(feats_gen, features_gen_bank)
        _, idxs_gen = distances_gen.sort()
        idxs_gen = idxs_gen[:, : args.num_neighbors]
        for i, (idx_src, idx_gen) in enumerate(zip(idxs_src, idxs_gen)):
            idx_common = idx_gen[(idx_src.unsqueeze(0) == idx_gen.unsqueeze(1)).nonzero()[:,0]][:args.num_near_neighbors].sort().values
            if len(idx_common) != 0:
                idxs[i, (args.num_near_neighbors-len(idx_common)):] = idx_common
                avg_nn += len(idx_common)

        # (64, num_nbrs, num_classes), average over dim=1
        probs = probs_bank[idxs, :].mean(1)
        pred_probs.append(probs)
        # (64, num_nbrs, num_classes)
        probs_all = probs_bank[idxs, :]
        pred_probs_all.append(probs_all)

    pred_probs_all = torch.cat(pred_probs_all)
    pred_probs = torch.cat(pred_probs)
    
    _, pred_labels = pred_probs.max(dim=1)
    # (64, num_nbrs, num_classes), max over dim=2
    _, pred_labels_all = pred_probs_all.max(dim=2)
    #First keep maximum for all classes between neighbors and then keep max between classes
    _, pred_labels_hard = pred_probs_all.max(dim=1)[0].max(dim=1)

    return pred_labels, pred_probs, pred_labels_all, pred_labels_hard


@torch.no_grad()
def soft_k_nearest_neighbors_alt(features, features_gen, features_bank, features_gen_bank, probs_bank, labels, labels_bank):
    pred_probs = []
    pred_probs_all = []
    avg_nn = 0
    
    for feats, feats_gen in zip(features.split(64), features_gen.split(64)):
        distances = get_distances(feats, features_bank)
        _, idxs_src = distances.sort()
        idxs = idxs_src[:, : args.num_near_neighbors]
        idxs_src = idxs_src[:, : args.num_neighbors]
        distances_gen = get_distances(feats_gen, features_gen_bank)
        _, idxs_gen = distances_gen.sort()
        idxs_alt = idxs_gen[:, : args.num_near_neighbors]
        idxs_gen = idxs_gen[:, : args.num_neighbors]
        for i, (idx_src, idx_gen) in enumerate(zip(idxs_src, idxs_gen)):
            idx_common = idx_gen[(idx_src.unsqueeze(0) == idx_gen.unsqueeze(1)).nonzero()[:,0]][:args.num_near_neighbors].sort().values
            if len(idx_common) != 0:
                idxs[i, (args.num_near_neighbors-len(idx_common)):] = idx_common
                idxs_alt[i, (args.num_near_neighbors-len(idx_common)):] = idx_common
                avg_nn += len(idx_common)
            
        # (64, num_nbrs, num_classes), average over dim=1
        probs = probs_bank[idxs_alt, :].mean(1)
        pred_probs.append(probs)
        # (64, num_nbrs, num_classes)
        probs_all = probs_bank[idxs_alt, :]
        pred_probs_all.append(probs_all)

    pred_probs_all = torch.cat(pred_probs_all)
    pred_probs = torch.cat(pred_probs)
    
    _, pred_labels = pred_probs.max(dim=1)
    # (64, num_nbrs, num_classes), max over dim=2
    _, pred_labels_all = pred_probs_all.max(dim=2)
    #First keep maximum for all classes between neighbors and then keep max between classes
    _, pred_labels_hard = pred_probs_all.max(dim=1)[0].max(dim=1)

    return pred_labels, pred_probs, pred_labels_all, pred_labels_hard


@torch.no_grad()
def soft_k_nearest_neighbors_threshold(features, features_gen, features_bank, features_gen_bank, probs_bank, labels, labels_bank):
    pred_probs = []
    pred_probs_all = []
    avg_nn = 0
    
    for feats, feats_gen in zip(features.split(64), features_gen.split(64)):
        distances = get_distances(feats, features_bank)
        _, idxs_src = distances.sort()
        idxs = idxs_src[:, : args.num_near_neighbors]
        distances_src = distances.clone()
        distances_gen = get_distances(feats_gen, features_gen_bank)
        _, idxs_gen = distances_gen.sort()
        for i in range(64):
            gradients = torch.gradient(distances_src[i][idxs_src[i]])[0]
            slope = (gradients < 0.01).nonzero()[0]
            idx_src = idxs_src[i, : int(slope)]
            gradients = torch.gradient(distances_gen[i][idxs_gen[i]])[0]
            slope = (gradients < 0.01).nonzero()[0]
            idx_gen = idxs_gen[i, : int(slope)]
            idx_common = idx_gen[(idx_src.unsqueeze(0) == idx_gen.unsqueeze(1)).nonzero()[:,0]][:args.num_near_neighbors].sort().values
            if len(idx_common) != 0:
                idxs[i, (args.num_near_neighbors-len(idx_common)):] = idx_common
                avg_nn += len(idx_common)
            
        # (64, num_nbrs, num_classes), average over dim=1
        probs = probs_bank[idxs, :].mean(1)
        pred_probs.append(probs)
        # (64, num_nbrs, num_classes)
        probs_all = probs_bank[idxs, :]
        pred_probs_all.append(probs_all)

    pred_probs_all = torch.cat(pred_probs_all)
    pred_probs = torch.cat(pred_probs)
    
    _, pred_labels = pred_probs.max(dim=1)
    # (64, num_nbrs, num_classes), max over dim=2
    _, pred_labels_all = pred_probs_all.max(dim=2)
    #First keep maximum for all classes between neighbors and then keep max between classes
    _, pred_labels_hard = pred_probs_all.max(dim=1)[0].max(dim=1)

    return pred_labels, pred_probs, pred_labels_all, pred_labels_hard


@torch.no_grad()
def soft_k_nearest_neighbors_src(features, features_gen, features_bank, features_gen_bank, probs_bank, labels, labels_bank):
    pred_probs = []
    pred_probs_all = []
    avg_nn = 0
    
    for feats, feats_gen in zip(features.split(64), features_gen.split(64)):
        distances = get_distances(feats, features_bank)
        _, idxs_src = distances.sort()
        idxs = idxs_src[:, : args.num_near_neighbors]
            
        # (64, num_nbrs, num_classes), average over dim=1
        probs = probs_bank[idxs, :].mean(1)
        pred_probs.append(probs)
        # (64, num_nbrs, num_classes)
        probs_all = probs_bank[idxs, :]
        pred_probs_all.append(probs_all)

    pred_probs_all = torch.cat(pred_probs_all)
    pred_probs = torch.cat(pred_probs)
    
    _, pred_labels = pred_probs.max(dim=1)
    # (64, num_nbrs, num_classes), max over dim=2
    _, pred_labels_all = pred_probs_all.max(dim=2)
    #First keep maximum for all classes between neighbors and then keep max between classes
    _, pred_labels_hard = pred_probs_all.max(dim=1)[0].max(dim=1)

    return pred_labels, pred_probs, pred_labels_all, pred_labels_hard


@torch.no_grad()
def soft_k_nearest_neighbors_gen(features, features_gen, features_bank, features_gen_bank, probs_bank, labels, labels_bank):
    pred_probs = []
    pred_probs_all = []
    avg_nn = 0
    
    for feats, feats_gen in zip(features.split(64), features_gen.split(64)):
        distances_gen = get_distances(feats_gen, features_gen_bank)
        _, idxs_gen = distances_gen.sort()
        idxs = idxs_gen[:, : args.num_near_neighbors]
            
        # (64, num_nbrs, num_classes), average over dim=1
        probs = probs_bank[idxs, :].mean(1)
        pred_probs.append(probs)
        # (64, num_nbrs, num_classes)
        probs_all = probs_bank[idxs, :]
        pred_probs_all.append(probs_all)

    pred_probs_all = torch.cat(pred_probs_all)
    pred_probs = torch.cat(pred_probs)
    
    _, pred_labels = pred_probs.max(dim=1)
    # (64, num_nbrs, num_classes), max over dim=2
    _, pred_labels_all = pred_probs_all.max(dim=2)
    #First keep maximum for all classes between neighbors and then keep max between classes
    _, pred_labels_hard = pred_probs_all.max(dim=1)[0].max(dim=1)

    return pred_labels, pred_probs, pred_labels_all, pred_labels_hard


def refine_predictions(
    feat,
    feat_gen,
    probs,
    labels,
    banks):
    
    feat_bank = banks["features"]
    feat_gen_bank = banks["features_gen"]
    probs_bank = banks["probs"]
    labels_bank = banks["labels"]
    
    pred_labels, probs, pred_labels_all, pred_labels_hard = soft_k_nearest_neighbors(
        feat, feat_gen, feat_bank, feat_gen_bank, probs_bank, labels, labels_bank
    )

    return pred_labels, probs, pred_labels_all, pred_labels_hard


def contrastive_loss(logits_ins, pseudo_labels, mem_labels):
    # labels: positive key indicators
    labels_ins = torch.zeros(logits_ins.shape[0], dtype=torch.long).cuda()

    mask = torch.ones_like(logits_ins, dtype=torch.bool)
    mask[:, 1:] = torch.all(pseudo_labels.unsqueeze(1) != mem_labels.unsqueeze(0), dim=2) 
    logits_ins = torch.where(mask, logits_ins, torch.tensor([float("-inf")]).cuda())

    loss = F.cross_entropy(logits_ins, labels_ins)

    return loss


@torch.no_grad()
def update_labels(banks, idxs, features_src, logits_src, features_gen):
    probs_src = F.softmax(logits_src, dim=1)

    start = banks["ptr"]
    end = start + len(idxs)
    idxs_replace = torch.arange(start, end).cuda() % len(banks["features"])
    banks["features"][idxs_replace, :] = features_src
    banks["probs"][idxs_replace, :] = probs_src
    banks["features_gen"][idxs_replace, :] = features_gen
    banks["ptr"] = end % len(banks["features"])

def div(logits, epsilon=1e-8):
    probs = F.softmax(logits, dim=1)
    probs_mean = probs.mean(dim=0)
    loss_div = -torch.sum(-probs_mean * torch.log(probs_mean + epsilon))

    return loss_div

def nl_criterion(output, y):
    output = torch.log( torch.clamp(1.-F.softmax(output, dim=1), min=1e-5, max=1.) )
    
    labels_neg = ( (y.unsqueeze(-1).repeat(1, 1) + torch.LongTensor(len(y), 1).random_(1, args.num_class).cuda()) % args.num_class ).view(-1)

    l = F.nll_loss(output, labels_neg, reduction='none')

    return l

# Training
def train(epoch, src_model, moco_model, gen_model, optimizer, trainloader, banks):
    
    loss = 0
    acc = 0

    src_model.train()
    moco_model.train()
    gen_model.eval()

    for batch_idx, batch in enumerate(trainloader): 
        weak_x = batch[0].cuda()
        strong_x = batch[1].cuda()
        y = batch[2].cuda()
        idxs = batch[3].cuda()
        strong_x2 = batch[5].cuda()

        feats_w, logits_w = moco_model(weak_x, cls_only=True)
        feats_gen = gen_model(weak_x).last_hidden_state.flatten(start_dim=1)

        if args.label_refinement:
            with torch.no_grad():
                probs_w = F.softmax(logits_w, dim=1)
                pseudo_labels_w, probs_w, _, _ = refine_predictions(feats_w, feats_gen, probs_w, y, banks)
        else:
            probs_w = F.softmax(logits_w, dim=1)
            pseudo_labels_w = probs_w.max(1)[1]
        
        _, logits_q, logits_ctr, keys = moco_model(strong_x, strong_x2)

        if args.ctr:
            loss_ctr = contrastive_loss(
                logits_ins=logits_ctr,
                pseudo_labels=moco_model.mem_labels[idxs],
                mem_labels=moco_model.mem_labels[moco_model.idxs]
            )
        else:
            loss_ctr = 0
        
        # update key features and corresponding pseudo labels
        moco_model.update_memory(epoch-1, idxs, keys, pseudo_labels_w, y)

        with torch.no_grad():
            #CE weights
            max_entropy = torch.log2(torch.tensor(args.num_class))
            w = entropy(probs_w)

            w = w / max_entropy
            w = torch.exp(-w)
        
        #Standard positive learning
        if args.neg_l:
            #Standard negative learning
            loss_cls = ( nl_criterion(logits_q, pseudo_labels_w)).mean()
            if args.reweighting:
                loss_cls = (w * nl_criterion(logits_q, pseudo_labels_w)).mean()
        else:
            loss_cls = ( CE(logits_q, pseudo_labels_w)).mean()
            if args.reweighting:
                loss_cls = (w * CE(logits_q, pseudo_labels_w)).mean()

        loss_div = div(logits_w) + div(logits_q)

        l = loss_cls + loss_ctr + loss_div

        update_labels(banks, idxs, feats_w, logits_w, feats_gen)

        l.backward()
        optimizer.step()
        optimizer.zero_grad()

        accuracy = 100.*accuracy_score(y.to('cpu'), logits_w.to('cpu').max(1)[1])

        loss += l.item()
        acc += accuracy
        
        if batch_idx % 100 == 0:
            print('Epoch [%3d/%3d] Iter[%3d/%3d]\t ' 
                    %(epoch, args.num_epochs, batch_idx+1, len(trainloader)))

            print("Acc ", acc/(batch_idx+1))

    
    print("Training acc = ", acc/len(trainloader))

    if args.wandb:
        wandb.log({
        'train_loss': loss_cls/len(trainloader), \
        'train_acc': acc/len(trainloader), \
        }, step=epoch)
        
    return loss_cls.item()/len(trainloader), acc/len(trainloader)


@torch.no_grad()
def eval_and_label_dataset(epoch, model, gen_model, banks):
    
    model.eval()
    gen_model.eval()
    logits, indices, gt_labels = [], [], []
    features = []
    features_gen = []

    for batch_idx, batch in enumerate(test_loader):
        inputs, targets, idxs = batch[0].cuda(), batch[2].cuda(), batch[3].cuda()
        
        feats, logits_cls = model(inputs, cls_only=True)
        feats_gen = gen_model(inputs).last_hidden_state.flatten(start_dim=1)

        features.append(feats)
        features_gen.append(feats_gen)
        gt_labels.append(targets)
        logits.append(logits_cls)
        indices.append(idxs)            

    features = torch.cat(features)
    features_gen = torch.cat(features_gen)
    gt_labels = torch.cat(gt_labels)
    logits = torch.cat(logits)
    indices = torch.cat(indices)
    
    probs = F.softmax(logits, dim=1)
    rand_idxs = torch.randperm(len(features)).cuda()
    banks = {
        "features": features[rand_idxs][: 16384],
        "features_gen": features_gen[rand_idxs][: 16384],
        "probs": probs[rand_idxs][: 16384],
        "labels": gt_labels[rand_idxs][: 16384],
        "ptr": 0,
    }

    # refine predicted labels
    pred_labels, _, _, _ = refine_predictions(features, features_gen, probs, gt_labels, banks)

    gt_labels = gt_labels.to('cpu')
    pred_labels = pred_labels.to('cpu')
    acc = 100.*accuracy_score(gt_labels, pred_labels)
    acc_class = 100.*accuracy_class(gt_labels, pred_labels)
        
    print("Evaluating target model!")

    print("\n| Test Epoch #%d\t Accuracy: %.2f%%\n" %(epoch, acc_class))  

    if args.wandb:
        wandb.log({'val_accuracy': acc_class}, step=epoch)

    return acc, acc_class, banks, gt_labels.numpy(), pred_labels.numpy()


def create_model(arch, args, pretrained):
    model = Resnet(arch, args.num_class, pretrained)

    model = model.cuda()
    return model


def accuracy_class(labels, preds):
    acc_class = np.zeros(args.num_class)
    total = np.zeros(args.num_class)
    for l,p in zip(labels, preds):
        if l == p:
            acc_class[l] += 1
        total[l] += 1
    
    return np.mean(np.divide(acc_class, total))


# Main code starts here

dataset_name = args.dataset.split('/')[0]

if dataset_name == 'officehome':
    if args.source[-13:] == "no-imbalanced":
        imbalanced = None
    else:
        imbalanced = "_UT"
elif dataset_name == 'visdac':
    if args.source[-2:] == "10":
        imbalanced = "10"
    elif args.source[-2:] == "50":
        imbalanced = "50"
    elif args.source[-3:] == "100":
        imbalanced = "100"
    else:
        imbalanced = None
elif dataset_name == 'domainnet':
    if args.source[-3:] == "126":
        imbalanced = "126"
    elif args.source[-4:] == "mini":
        imbalanced = "mini"
    else:
        print("ERROR: Unknown config for domainnet %s" % imbalanced)
        exit()
else:
    print("ERROR: Unknown dataset %s" % dataset_name)
    exit()
        
arch = 'resnet18'
num_workers = 4

if dataset_name == 'pacs':
    train_dataset = dataset(dataset=args.dataset, root=join(args.data_dir, 'PACS'), imb=imbalanced, noisy_path=None,
                          mode='all',
                          transform=transforms.Compose([transforms.Resize(256), transforms.RandomCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
                          )

    test_dataset = dataset(dataset=args.dataset, root=join(args.data_dir, 'PACS'), imb=imbalanced, noisy_path=None,
                         mode='all',
                         transform=transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
                         )
    
if dataset_name == 'officehome':
    train_dataset = dataset(dataset=args.dataset, root=join(args.data_dir, 'officeHome'), imb=imbalanced, noisy_path=None,
                          mode='train',
                          transform=transforms.Compose([transforms.Resize(256), transforms.RandomCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
                          )

    test_dataset = dataset(dataset=args.dataset, root=join(args.data_dir, 'officeHome'), imb=imbalanced, noisy_path=None,
                         mode='train',
                         transform=transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
                         )
    
    arch = 'resnet50'
    num_workers = 2

elif dataset_name == 'visdac':
    train_dataset = dataset(dataset=args.dataset, root=join(args.data_dir, 'VISDA'), imb=imbalanced, noisy_path=None,
                          mode='train',
                          transform=transforms.Compose([transforms.Resize(256), transforms.RandomCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
                          )

    test_dataset = dataset(dataset=args.dataset, root=join(args.data_dir, 'VISDA'), imb=imbalanced, noisy_path=None,
                         mode='test',
                         transform=transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
                         )
    
    arch = 'resnet101'

elif dataset_name == 'domainnet':
    train_dataset = dataset(dataset=args.dataset, root=join(args.data_dir, 'domainNet'), imb=imbalanced, noisy_path=None,
                          mode='train',
                          transform=transforms.Compose([transforms.Resize(256), transforms.RandomCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
                          )
    
    test_dataset = dataset(dataset=args.dataset, root=join(args.data_dir, 'domainNet'), imb=imbalanced, noisy_path=None,
                         mode='test',
                         transform=transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
                         )
    
    arch = 'resnet50'
    num_workers = 2

logdir = join("logs", args.run_name)
src_model = create_model(arch, args, pretrained=None)
gen_model = SwinModel.from_pretrained("microsoft/swin-base-patch4-window7-224-in22k").cuda()
momentum_model = create_model(arch, args, pretrained=None)

load_weights(src_model, 'logs/' + args.source + '/weights_best.tar')
load_weights(momentum_model, 'logs/' + args.source + '/weights_best.tar')

optimizer_src = optim.SGD(src_model.parameters(), lr=args.lr, weight_decay=5e-4)

moco_model = AdaMoCo(src_model = src_model, momentum_model = momentum_model, features_length=src_model.bottleneck_dim, num_classes=args.num_class, dataset_length=len(train_dataset), temporal_length=args.temporal_length)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

cudnn.benchmark = True

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=args.batch_size,
                                               num_workers=num_workers,
                                               drop_last=True,
                                               shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=args.batch_size,
                                              num_workers=num_workers,
                                              drop_last=True,
                                              shuffle=False)

CE = nn.CrossEntropyLoss(reduction='none')
CEloss = nn.CrossEntropyLoss()

makedirs(logdir, exist_ok=True)

if args.wandb:
    wandb.log({
    'train_loss': 0, \
    'train_acc': 0, \
    }, step=0)
best_train = 0
best_val = 0
best_val_class = 0

val_acc, val_acc_class, banks, _, _ = eval_and_label_dataset(0, moco_model, gen_model, None)
loss_train = [0]
accuracy_train = [0]
accuracy_val = [val_acc]
accuracy_val_class = [val_acc_class]

print("Training started!")
for epoch in range(1, args.num_epochs+1):
    start_time = time.time()
    train_loss, train_acc = train(epoch, src_model, moco_model, gen_model, optimizer_src, train_loader, banks) # train network
    
    val_acc, val_acc_class, banks, gt_labels, pred_labels = eval_and_label_dataset(epoch, moco_model, gen_model, banks)
    print("Time elapsed = %.1fs" % (time.time() - start_time))
    loss_train.append(train_loss)
    accuracy_train.append(train_acc)
    accuracy_val.append(val_acc)
    accuracy_val_class.append(val_acc_class)
    with open(join(logdir, "loss_train.pkl"), "wb") as fp:  # Pickling
        pickle.dump(loss_train, fp)
    with open(join(logdir, "accuracy_train.pkl"), "wb") as fp:  # Pickling
        pickle.dump(accuracy_train, fp)
    with open(join(logdir, "accuracy_val.pkl"), "wb") as fp:  # Pickling
        pickle.dump(accuracy_val, fp)
    with open(join(logdir, "accuracy_val_class.pkl"), "wb") as fp:  # Pickling
        pickle.dump(accuracy_val_class, fp)

    if train_acc > best_train:
        save_weights(src_model, epoch, logdir + '/weights_best_train.tar')
        with open(join(logdir, "gt_labels_train.pkl"), "wb") as fp:  # Pickling
            pickle.dump(gt_labels, fp)
        with open(join(logdir, "pred_labels_train.pkl"), "wb") as fp:  # Pickling
            pickle.dump(pred_labels, fp)
        best_train = train_acc
        
    if val_acc > best_val:
        save_weights(src_model, epoch, logdir + '/weights_best_val.tar')
        with open(join(logdir, "gt_labels_val.pkl"), "wb") as fp:  # Pickling
            pickle.dump(gt_labels, fp)
        with open(join(logdir, "pred_labels_val.pkl"), "wb") as fp:  # Pickling
            pickle.dump(pred_labels, fp)
        best_val = val_acc
        
    if val_acc_class > best_val_class:
        save_weights(src_model, epoch, logdir + '/weights_best_val_class.tar')
        with open(join(logdir, "gt_labels_val_class.pkl"), "wb") as fp:  # Pickling
            pickle.dump(gt_labels, fp)
        with open(join(logdir, "pred_labels_val_class.pkl"), "wb") as fp:  # Pickling
            pickle.dump(pred_labels, fp)
        best_val_class = val_acc_class

        if args.wandb:
            wandb.run.summary['best_acc'] = best_train

print("\n *** Validation accuracy (class_wise) for the best adaptation: %.2f *** " % best_val_class)

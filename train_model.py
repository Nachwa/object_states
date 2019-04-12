import json, argparse
import numpy as np
import time
from tqdm import tqdm
import model
from epic_db import EPIC_Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from collections import Counter
import torch
import torch.nn as nn

np.random.seed(2222)
torch.manual_seed(2222)
#---------------------------------------------------------------------------------------------------

def freeze_backbone(freeze):
    for child in net.children():
        for param in child.parameters():
            param.requires_grad = True  
    for child in net.features_layers.children():
        for param in child.parameters():
            param.requires_grad = not freeze

#---------------------------------------------------------------------------------------------------
def compute_loss(pred, true, fn='ce'):  
    if   fn == 'mse':
        loss_fn   = nn.MSELoss(reduce=True)
    elif fn == 'ce':
        loss_fn    = nn.CrossEntropyLoss()
    elif fn == 'bce':
        loss_fn   = nn.BCEWithLogitsLoss()
    elif fn == 'hinge':
        loss_fn = nn.MultiLabelMarginLoss()
    return  loss_fn(pred, true) 
#---------------------------------------------------------------------------------------------------
def count_metrics(predictions, targets, n_classes):

    tps, fps, fns = np.zeros(n_classes), np.zeros(n_classes), np.zeros(n_classes)
    for i in range(n_classes):
        pred_i = (predictions == i)
        true_i = (targets == i)
        
        tps[i] = np.sum(targets[pred_i]==i)
        fps[i] = np.sum(pred_i) - tps[i]
        fns[i] = np.sum(true_i) - tps[i] 

    nb_samples = predictions.shape[0]
    micro_acc = np.sum(predictions == targets) 

    return Counter({'TP': tps, 'FP': fps, 'FN': fns, 
                    'micro_acc'    : micro_acc, 'nb_samples'  : nb_samples})
       
#---------------------------------------------------------------------------------------------------
def calc_metrics(counter, ignore_classes = []):
    
    tps = counter['TP']
    fns = counter['FN']
    fps = counter['FP']
    
    n_classes    = tps.shape[0]
    mask_classes = np.ones(n_classes, dtype=bool)
    mask_classes[ignore_classes] = False

    tps = tps[mask_classes]
    fns = fns[mask_classes]
    fps = fps[mask_classes]

    precision = np.mean(np.divide(tps, tps + fps, out=np.zeros_like(tps), where=(tps + fps)!=0)) 
    recall    = np.mean(np.divide(tps, tps + fns, out=np.zeros_like(tps), where=(tps + fns)!=0)) 
    acc       = counter['micro_acc'] / counter['nb_samples']

    return acc, precision, recall

#---------------------------------------------------------------------------------------------------
def iteration_step(epoch, update_weights=True):
    #set model in training mode
    if update_weights: 
        net.train()
        loader = train_img_loader
        tabname = 'train' 
    else:
        net.eval()
        loader = valid_img_loader
        tabname = 'valid'
    
    #prepare evaluation metrics
    loss_lst = []
    
    noun_counter, verb_counter, action_counter = Counter(), Counter(), Counter()

    for i, data in enumerate(tqdm(loader)):
        verb_class_many = data['verb_class_many']
        all_nouns_many  = data['all_nouns_many']
        all_nouns       = data['all_nouns']
        epic_states     = data['obj_states']
        epic_nouns      = data['noun_class']
        epic_verbs      = data['verb_class']
        epic_action     = data['action_class']
        rgb_frame       = data['rgb_frame']
        
        rgb_frame = rgb_frame.cuda()

        #pass img to the model
        (out11, out12), (out21, out22), (out31, out32), (out41, out42) = net(rgb_frame)
        
        loss = compute_loss(out31, epic_verbs.cuda(), fn='ce') + \
               compute_loss(out32, epic_nouns.cuda(), fn='ce')  + \
               compute_loss(out41, epic_action.cuda(), fn='ce') 

        for i in range(epic.nb_keyframes):
            loss +=  compute_loss(out12[:,i].view(-1, epic.n_many_nouns), all_nouns_many.cuda(), fn='mse') + \
                     compute_loss(out22[:,i].view(-1, epic.n_states), epic_states[:, i].cuda(), fn='mse') + \
                     compute_loss(out11[:,i].view(-1, epic.n_nouns), all_nouns.cuda(), fn='mse')
         
        if update_weights:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        #compute accuracy 
        loss_lst.append(loss.data.item())
        
        max_pred_verb = torch.argmax(out31, dim=-1).long().cpu().data.numpy()
        max_pred_noun = torch.argmax(out32, dim=-1).long().cpu().data.numpy()
        max_pred_action = torch.argmax(out41, dim=-1).long().cpu().data.numpy()
        
        noun_counter.update(count_metrics(max_pred_noun, epic_nouns.numpy(), epic.n_nouns))
        verb_counter.update(count_metrics(max_pred_verb, epic_verbs.numpy(), epic.n_verbs))
        action_counter.update(count_metrics(max_pred_action, epic_action.numpy(), epic.n_actions))

    loss_mean = np.mean(loss_lst)

    noun_acc, noun_precision, noun_recall = calc_metrics(dict(noun_counter), ignore_classes=epic.noun_ignore_cls)
    verb_acc, verb_precision, verb_recall = calc_metrics(dict(verb_counter), ignore_classes=epic.verb_ignore_cls) 
    action_acc, action_precision, action_recall = calc_metrics(dict(action_counter), ignore_classes=epic.action_ignore_cls) 


    return loss_mean, (verb_acc, noun_acc, action_acc), (verb_precision, noun_precision, action_precision), (verb_recall, noun_recall, action_recall)
#---------------------------------------------------------------------------------------------------
def save_checkpoint(state, filename):
    """Save checkpoint if a new best is achieved"""
    torch.save(state, filename)  # save checkpoint
#---------------------------------------------------------------------------------------------------
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', dest='cuda', type=bool,
                        default=True, help='use cuda')
    parser.add_argument('--logname', dest='tb_logname', type=str,
                        help='name of experiment for tensorboard')
    parser.add_argument('--checkpoint', dest='checkpoint', type=str, 
                        default='', help='resume training from a checkpoint')
    parser.add_argument('--epochs', dest='epochs', type=int, 
                        default=100, help='number of epochs to run')
    parser.add_argument('--save_checkpoint', dest='save_checkpoint', type=bool, 
                        default=True, help='to save or not checkpoints files')
    parser.add_argument('--batch_size', dest='batch_size', type=int, 
                        default=32, help='batch size')
    parser.add_argument('--model_name', dest='model_name', type=str, 
                        help='one of the following obj_cam') 
    parser.add_argument('--db_dir', dest='db_dir', type=str, 
                        default='/home/naboubak/Downloads/EPIC_KITCHENS_2018', help='root directory of dataset')
    parser.add_argument('--nb_keyframes', dest='nb_keyframes', type=int, 
                        default=5, help='number of keyframes')
    return parser.parse_args()

#---------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    args = get_args()

    work_db_dir = args.db_dir
    epic = EPIC_Dataset(work_db_dir, dataset='train', nb_keyframes=args.nb_keyframes, with_metadata=True)

    with open('./input_data/indices_new.txt', 'r') as file:
        db_indices = eval(file.readline())

    fold_train = db_indices[:int(len(epic) * 0.8)]
    fold_valid = db_indices[int(len(epic) * 0.8):]

    train_sampler = SubsetRandomSampler(fold_train)
    valid_sampler = SubsetRandomSampler(fold_valid)
    
    train_img_loader = torch.utils.data.DataLoader(epic, 
                                                batch_size=args.batch_size, 
                                                num_workers=32, 
                                                sampler=train_sampler)

    valid_img_loader = torch.utils.data.DataLoader(epic,
                                                sampler=valid_sampler,
                                                batch_size=args.batch_size, 
                                                num_workers=32)
    
    print('Initialize network architecture .. ')
    net = model.action_noun_state_cam(epic.n_many_nouns, epic.n_states, epic.n_many_verbs, nb_keyframes = args.nb_keyframes).cuda()
    print(net)

    epoch_n, best_score = -1, 0.0
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)
        epoch_n = checkpoint['epoch']
        args.cuda = checkpoint['arch']
        best_score = checkpoint['best_score']
        net_arch = checkpoint['net_arch']
        
        model_state_dict = net.state_dict()
        pretrained_state_dict = checkpoint['state_dict']
        pretrained_state_dict = {k: v for k, v in pretrained_state_dict.items() if k in model_state_dict}

        net.load_state_dict(pretrained_state_dict)
        
        args.tb_logname = checkpoint['model_name'] if args.tb_logname is None else args.tb_logname
        print("=> loaded checkpoint '{}' of model {} at (epoch {})"
                          .format(args.checkpoint, checkpoint['model_name'], checkpoint['epoch']))
        
    
    freeze_backbone(True)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,net.parameters()))
    print('number of trainable paramerters:',  sum(p.numel() for p in net.parameters() if p.requires_grad))
    print('number of all paramerters:',  sum(p.numel() for p in net.parameters()))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, verbose=True) 
    
    epoch_n += 1
    print('Start training .. ')
    from datetime import datetime
    for epoch in range(args.epochs):
        log_txt = f'{epoch:02d} \t'
        
        train_loss, tr_acc_scores, tr_precision_scores, tr_recall_scores = iteration_step(epoch = epoch_n, update_weights=True)   
        print(f'epoch ({epoch_n:2d}): train [loss = {train_loss:.03f}]', end=', ')
        print(f'acc [v: {tr_acc_scores[0]*100:2.02f}, n: {tr_acc_scores[1]*100:2.02f}, a: {tr_acc_scores[2]*100:2.02f}]', end = ', ')
        print(f'precision [v: {tr_precision_scores[0]*100:2.02f}, n: {tr_precision_scores[1]*100:2.02f}, a: {tr_precision_scores[2]*100:2.02f}]', end = ', ')
        print(f'recall [v: {tr_recall_scores[0]*100:2.02f}, n: {tr_recall_scores[1]*100:2.02f}, a: {tr_recall_scores[2]*100:2.02f}]')
        log_txt += f'{datetime.now()}: train [loss = {train_loss:.03f}], acc [v: {tr_acc_scores[0]*100:2.02f}, n: {tr_acc_scores[1]*100:2.02f}, a: {tr_acc_scores[2]*100:2.02f}], precision [v: {tr_precision_scores[0]*100:2.02f}, n: {tr_precision_scores[1]*100:2.02f}, a: {tr_precision_scores[2]*100:2.02f}], recall [v: {tr_recall_scores[0]*100:2.02f}, n: {tr_recall_scores[1]*100:2.02f}, a: {tr_recall_scores[2]*100:2.02f}] \n'

        valid_loss, vl_acc_scores, vl_precision_scores, vl_recall_scores = iteration_step(epoch = epoch_n, update_weights=False)
        scheduler.step(valid_loss)

        v_F1_score = (3 * vl_precision_scores[0] * vl_recall_scores[0] * vl_acc_scores[0]) / (vl_precision_scores[0] + vl_recall_scores[0] + vl_acc_scores[0])
        n_F1_score = (3 * vl_precision_scores[1] * vl_recall_scores[1] * vl_acc_scores[1]) / (vl_precision_scores[1] + vl_recall_scores[1] + vl_acc_scores[1])
        F1_score = 100 * (v_F1_score + n_F1_score) / 2.0

        print(f'valid [loss = {valid_loss:.03f}]', end=', ')
        print(f'acc [v: {vl_acc_scores[0]*100:2.02f}, n: {vl_acc_scores[1]*100:2.02f}, a: {vl_acc_scores[2]*100:2.02f}]', end = ', ')
        print(f'precision [v: {vl_precision_scores[0]*100:2.02f}, n: {vl_precision_scores[1]*100:2.02f}, a: {vl_precision_scores[2]*100:2.02f}]', end = ', ')
        print(f'recall [v: {vl_recall_scores[0]*100:2.02f}, n: {vl_recall_scores[1]*100:2.02f}, a: {vl_recall_scores[2]*100:2.02f}], [my_F1: {F1_score:2.03f}]')
        log_txt += f'{datetime.now()}: valid [loss = {valid_loss:.03f}], acc [v: {vl_acc_scores[0]*100:2.02f}, n: {vl_acc_scores[1]*100:2.02f}, a: {vl_acc_scores[2]*100:2.02f}], precision [v: {vl_precision_scores[0]*100:2.02f}, n: {vl_precision_scores[1]*100:2.02f}, a: {vl_precision_scores[2]*100:2.02f}], recall [v: {vl_recall_scores[0]*100:2.02f}, n: {vl_recall_scores[1]*100:2.02f}, a: {vl_recall_scores[2]*100:2.02f}], [my_F1: {F1_score:2.03f}] \n'
             
        if args.save_checkpoint:
            if best_score <  F1_score: 
                best_score = F1_score
                save_checkpoint({
                        'epoch'      : epoch_n,
                        'arch'       : args.cuda,
                        'state_dict' : net.state_dict(),
                        'net_arch'   : str(net),
                        'best_score' : best_score,
                        'optimizer'  : optimizer.state_dict(),
                        'model_name' : args.tb_logname
                        }, filename=f'./checkpoints/{args.tb_logname}_{best_score:2.02f}_ep{epoch_n}.pth')

        epoch_n += 1

        with open(f'log_{args.tb_logname}.txt', 'a') as logfile:
            logfile.write(log_txt)
    

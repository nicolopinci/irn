import torch
from torch.backends import cudnn
cudnn.enabled = True
from torch.utils.data import DataLoader
import torch.nn.functional as F

import importlib

import voc12.dataloader
from misc import pyutils, torchutils


def validate(model, data_loader):
    print('validating ... ', flush=True, end='')

    val_loss_meter = pyutils.AverageMeter('loss1', 'loss2')

    model.eval()

    with torch.no_grad():
        for pack in data_loader:
            img = pack['img']

            label = pack['label'].cuda(non_blocking=True)

            x = model(img)
            loss1 = F.multilabel_soft_margin_loss(x, label)

            val_loss_meter.add({'loss1': loss1.item()})

    model.train()

    validation_loss = (val_loss_meter.pop('loss1'))
    
    print('loss: %.4f' % validation_loss)

    return validation_loss


def run(args):

    model = getattr(importlib.import_module(args.cam_network), 'Net')()


    train_dataset = voc12.dataloader.VOC12ClassificationDataset(args.train_list, voc12_root=args.voc12_root,
                                                                resize_long=(320, 640), hor_flip=True,
                                                                crop_size=512, crop_method="random")
    train_data_loader = DataLoader(train_dataset, batch_size=args.cam_batch_size,
                                   shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    max_step = (len(train_dataset) // args.cam_batch_size) * args.cam_num_epoches

    val_dataset = voc12.dataloader.VOC12ClassificationDataset(args.val_list, voc12_root=args.voc12_root,
                                                              crop_size=512)
    val_data_loader = DataLoader(val_dataset, batch_size=args.cam_batch_size,
                                 shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=True)

    param_groups = model.trainable_parameters()
    optimizer = torchutils.PolyOptimizer([
        {'params': param_groups[0], 'lr': args.cam_learning_rate, 'weight_decay': args.cam_weight_decay},
        {'params': param_groups[1], 'lr': 10*args.cam_learning_rate, 'weight_decay': args.cam_weight_decay},
    ], lr=args.cam_learning_rate, weight_decay=args.cam_weight_decay, max_step=max_step)
    

    model = torch.nn.DataParallel(model).cuda()
    model.train()

    avg_meter = pyutils.AverageMeter()

    timer = pyutils.Timer()
    
    new_validation_loss = float('inf')
    old_validation_loss = float('inf')
    optimal_validation_loss = float('inf')
    
    early_stop_now = False
    
    ep = 0
    ep_max = args.cam_num_epoches
    
    training_vec = []
   

    while(ep < ep_max and early_stop_now is False):

        old_validation_loss = new_validation_loss
        
        print('Epoch %d/%d' % (ep+1, args.cam_num_epoches))

        for step, pack in enumerate(train_data_loader):

            img = pack['img']
            label = pack['label'].cuda(non_blocking=True)

            x = model(img)
            loss = F.multilabel_soft_margin_loss(x, label)

            avg_meter.add({'loss1': loss.item()})
            
            current_train_loss = 0

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (optimizer.global_step-1)%100 == 0:
                timer.update_progress(optimizer.global_step / max_step)
                
                current_train_loss = avg_meter.pop('loss1')
                training_vec.append(current_train_loss)

                print('step:%5d/%5d' % (optimizer.global_step - 1, max_step),
                      'loss:%.4f' % (current_train_loss),
                      'imps:%.1f' % ((step + 1) * args.cam_batch_size / timer.get_stage_elapsed()),
                      'lr: %.4f' % (optimizer.param_groups[0]['lr']),
                      'etc:%s' % (timer.str_estimated_complete()), flush=True)

        else:
            new_validation_loss = validate(model, val_data_loader)
            timer.reset_stage()
        
        if(new_validation_loss < optimal_validation_loss):
            optimal_validation_loss = new_validation_loss
            
        GL_value = calculateGL(optimal_validation_loss, new_validation_loss)
        Pk_value = calculatePk(training_vec, args.stopping_k)
        PQ_value = calculatePQ(GL_value, Pk_value)
        
        print('GL:%.1f' % GL_value)
        print('P(k):%.2f' % Pk_value)
        print('PQ:%.2f' % PQ_value)
        
        if(args.stopping_criterion == "threshold" and GL_value > args.stopping_threshold):
            early_stop_now = True
            
        if(args.stopping_criterion == "strip" and PQ_value > args.stopping_threshold):
            early_stop_now = True
            
        if(args.stopping_criterion == "onlyPk" and 100/Pk_value > args.stopping_treshold):
            early_stop_now = True
            
        ep += 1 
        
        
        
    if(early_stop_now == True and ep < ep_max):
        print("Early stopping activated")
        

    torch.save(model.module.state_dict(), args.cam_weights_name + '.pth')
    torch.cuda.empty_cache()
    
    
def calculateGL(Eopt, Eva):
    return 100*(Eva/Eopt - 1)

def calculatePk(training_vec, k):
    num = 0.0
    den = float('inf')
    min_tv = float('inf')
    last_position = len(training_vec)
    initial_position = max(last_position - k, 0)
    
    for i in range(initial_position, last_position):
        num = num + training_vec[i]
        if(min_tv > training_vec[i]):
            min_tv = training_vec[i]
            
    den = k*min_tv
    
    return 1000*(num/den-1.0)


def calculatePQ(GL, Pk):
    return GL/Pk
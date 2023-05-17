from utils import *
from attn import *
from UNADON_model import *

import numpy as np
import math
import json

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import time



def train(conf):
    id = get_free_gpu()
    device = torch.device("cuda:%d" % id) 

    # Step 1: Load train/valid/test data as defined in the config file

    (X_train, y_train, d_train, coord_train) = load_data(conf['training_cell_type'], \
            conf['training_chr'], conf['train_data_path'], conf['feature'],\
            conf['window_size'], conf['y'], conf['histone'])
    print('Train set:', X_train.shape, y_train.shape, coord_train.shape)
    

    (X_valid, y_valid, d_valid, coord_valid) = load_data(conf['validation_cell_type'], \
            conf['validation_chr'], conf['valid_data_path'], conf['feature'],\
            conf['window_size'], conf['y'], conf['histone'])
    print('Valid set:', X_valid.shape, y_valid.shape, coord_valid.shape)

    (X_test, y_test, d_test, coord_test) = load_data(conf['testing_cell_type'],\
            conf['testing_chr'], conf['test_data_path'], conf['feature'],\
            conf['window_size'], conf['y'], conf['histone'])
    print('Test set:', X_test.shape, y_test.shape, coord_test.shape)


    dts_train = SeqData(X_train, y_train, d_train, coord_train)
    dts_valid = SeqData(X_valid, y_valid, d_valid, coord_valid)
    dts_test = SeqData(X_test, y_test, d_test, coord_test)

    # Only needed for cross-cell-type prediction when we want to assign different weight for different training cell types
    sampler = WeightedRandomSampler(dts_train.get_sample_weight(), len(dts_train) * 2)

    loader_train = DataLoader(dataset = dts_train, 
                              batch_size = conf['batch'],
                              pin_memory=True,
                              num_workers = 2,
                              # shuffle = True,
                              sampler = sampler
                              )

    loader_valid = DataLoader(dataset = dts_valid, 
                              batch_size = conf['batch'] ,
                              pin_memory=True,
                              num_workers = 2,
                              shuffle = False)


    loader_test = DataLoader(dataset = dts_test, 
                              batch_size = conf['batch'],
                              pin_memory=True,
                              num_workers = 2,
                              shuffle = False)


    input_dim = X_train.shape[-1]
    


    np.random.seed(conf['random_state'])
    torch.manual_seed(conf['random_state'])


    # Step 2: Prepare the model
    model = UNADON(input_dim, 20, conf['dense_dim'], conf['dense_num_layers'],\
    conf['nhead'],  conf['attn_hidden_dim'], conf['attn_layers'], conf["dropout"])
    # assume that the dimension of the sequence features is 20.

    model.to(device)
    conf['device'] = device


    print('Number of parameters',count_parameters(model))

    # Step 3: Define loss function and optimizer
    loss_func = nn.MSELoss()
    domain_loss_func = nn.CrossEntropyLoss() # For cross-cell-type only
    optimizer = optim.AdamW(model.parameters(),lr = conf['base_lr'], weight_decay = conf['base_lr'], betas = (0.9,0.98))
    warmup_steps = 3000
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, \
        lambda step: min((step+1)**(-0.5), (step+1)*(warmup_steps) ** (-1.5)))


    t = time.time()
    print('Start training...')
    best_corr = -1
    best_r = -1
    best_epoch = 0


    # Step 4: Start training
    for epoch in range(conf['num_of_epochs']):
        model.train()    
        loss_list = []
        domain_loss_list = []
        y_true_list = []
        y_prob_list = []
        coord_list = []
        print(epoch)

        for i, (X,y,d,coord) in enumerate(loader_train):
            (X, y, d) = (X.to(device), y.to(device), d.to(torch.long).to(device))
            coord = np.array(coord).T
            (outputs, extra) = model(X)
            prediction_loss = loss_func(outputs, y)

            domain_loss = domain_loss_func(extra, d)
            
            if conf['mode'] == 'Single':
                loss = prediction_loss 
            elif conf['mode'] == 'Cross':
                # Domain adaptation for cross-cell-type prediction
                loss = sum([prediction_loss, 2 * domain_loss])
            else:
                raise Exception('Experiment type not defined. Please use Single or Cross.')


            loss_list.append(loss.item())

            y_true = y.cpu().detach().numpy()
            y_true_list.extend(y_true)

            y_prob_list.extend(outputs.cpu().detach().numpy())
            coord_list.extend(coord)


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        y_true_list = np.concatenate(y_true_list)
        y_prob_list = np.concatenate(y_prob_list)
        coord_list = np.concatenate(coord_list)


        print('Epoch %d, training loss %.5f' % (epoch, np.mean(loss_list)))

        reg_report(y_true_list, y_prob_list)


        # on the validation cell type
        model.eval()

        loss_list = []
        y_true_list = []
        y_prob_list = []
        coord_list = []

        for i, (X,y,d,coord) in enumerate(loader_valid):
            (X, y) = (X.to(device), y.to(device))
            coord = np.array(coord).T
            (outputs,_) = model(X)

            loss = loss_func(outputs, y)
            loss_list.append(loss.item())
            
            y_true = y.cpu().detach().numpy()
            y_true_list.extend(y_true)
            
            y_prob_list.extend(outputs.cpu().detach().numpy())
            coord_list.extend(coord)

        coord_list = np.concatenate(coord_list)
        y_true_list = np.concatenate(y_true_list)
        y_prob_list = np.concatenate(y_prob_list)

    
        print('Valid loss %.5f' % np.mean(loss_list))

        metric = reg_report(y_true_list, y_prob_list)
        # loss = merge_pred(y_true_list, y_prob_list, coord_list,epoch,conf['output_path'],conf['output_name'],'valid')
        # print('Valid loss after merging %.5f' % loss)
        
        # Used both correlation and R2 to define the best epoch
        if metric[1] > best_corr and metric[3] > best_r:
            best_corr = max(metric[1], best_corr)
            best_r = max(metric[3], best_r)
            best_epoch = epoch

            if not os.path.exists(conf['output_path'] + 'model/'):
                os.mkdir(conf['output_path'] + 'model/')

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss
                }, conf['output_path'] + 'model/%s_%s_%s_best.pth' % (conf['mode'],conf['y'],conf['IML_cell_type']))
        else:
            print('Time for epoch: ', time.time() - t)
            t = time.time()
            # Only calculate test loss for the best models
            continue

        # if validation only (e.g. hyperparameter tuning), skip the testing step
        if not conf['run_test']:
            print('Time for epoch: ', time.time() - t)
            t = time.time()
            continue

        # Evaluate on testing cell type
        model.eval()

        loss_list = []
        y_true_list = []
        y_prob_list = []
        coord_list = []

        for i, (X,y,d,coord) in enumerate(loader_test):
            (X, y) = (X.to(device), y.to(device))
            coord = np.array(coord).T
            (outputs,_) = model(X)

            loss = loss_func(outputs, y)
            loss_list.append(loss.item())
            
            y_true = y.cpu().detach().numpy()
            y_true_list.extend(y_true)
            
            y_prob_list.extend(outputs.cpu().detach().numpy())
            coord_list.extend(coord)

        coord_list = np.concatenate(coord_list)
        y_true_list = np.concatenate(y_true_list)
        y_prob_list = np.concatenate(y_prob_list)
 

        print('Test loss %.5f' % np.mean(loss_list))

        
        metric = reg_report(y_true_list, y_prob_list)
        loss = merge_pred(y_true_list, y_prob_list, coord_list,epoch,conf['output_path'],conf['output_name'],'test')
        print('Test loss after merging %.5f' % loss)


        print('Time for epoch: ', time.time() - t)
        t = time.time()



if __name__ == '__main__':
    with open('../config/config.json', 'r') as f:
        conf = json.load(f)
    train(conf)

















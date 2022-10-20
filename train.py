import numpy as np
from utils import *
from UNADON_model import *
import math
import json
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import time




def train(model, loader_train, loader_test, device, \
          output_name):

    num_epochs = conf['num_of_epochs']
    loss_func = nn.MSELoss()
    domain_loss_func = nn.CrossEntropyLoss()
    epi_loss_func = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(),lr = conf['base_lr'], \
                            weight_decay=conf['reg'], betas = (0.9,0.98))

    warmup_steps = 3000
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda step: min((step+1)**(-0.5), (step+1)*(warmup_steps) ** (-1.5)))
    # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.97)

    t = time.time()
    print('Start training...')
    best_corr = -1
    best_r = -1
    best_epoch = 0

    for epoch in range(num_epochs):
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
            if conf['task'] == 'classification' and num_classes == 2:
                y = y.type_as(outputs)
                outputs = nn.Flatten(-2,-1)(outputs)
            prediction_loss = loss_func(outputs, y)
            domain_loss = domain_loss_func(extra, d)
            loss = prediction_loss 
            # loss = sum([prediction_loss, domain_loss])

            loss_list.append(loss.item())
            # domain_loss_list.append(domain_loss.item())
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

        print(y_true_list.shape, y_prob_list.shape)

        print('Epoch %d, training loss %.5f' % (epoch, np.mean(loss_list)))
        # print('Domain prediction loss', np.mean(domain_loss_list))

        reg_report(y_true_list, y_prob_list)


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


        print(y_true_list[:20].flatten())
        print(y_prob_list[:20].flatten())

        print('Test loss %.5f' % np.mean(loss_list))

        metric = reg_report(y_true_list, y_prob_list)
        if epoch >= conf['merge_epoch']:
            loss = merge_pred(y_true_list, y_prob_list, coord_list,epoch,output_name,'test')
            print('Test loss after merging %.5f' % loss)

        if metric[1] > best_corr or metric[3] > best_r:
            best_corr = max(metric[1], best_corr)
            best_r = max(metric[3], best_r)
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss
                }, 'result/model/%s_best.pth' % (output_name))


        print('Time for epoch: ', time.time() - t)
        t = time.time()





def main():
    id = get_free_gpu()
    device = torch.device("cuda:%d" % id) 
    # device = 'cuda:4'


    (X_train, y_train, d_train, coord_train) = load_data(conf['training_cell_type'], 
                                                         conf['training_chr'], 
                                                        conf['train_data_path'], 
                                                        conf['feature'],
                                                        conf['window_size'], 
                                                        conf['histone'])
    print('Training data',X_train.shape, y_train.shape, coord_train.shape)

    (X_test, y_test, d_test, coord_test) = load_data(conf['testing_cell_type'], 
                                                     conf['testing_chr'], 
                                                     conf['test_data_path'], 
                                                     conf['feature'],
                                                     conf['window_size'], 
                                                     conf['histone'])
    print('Evaluation data',X_test.shape, y_test.shape, coord_test.shape)

    dts_train = SeqData(X_train, y_train, d_train, coord_train)
    # dts_valid = SeqData(X_valid, y_valid, d_valid, coord_valid)
    dts_test = SeqData(X_test, y_test, d_test, coord_test)

    sampler = WeightedRandomSampler(dts_train.get_sample_weight(), len(dts_train) * 2)

    loader_train = DataLoader(dataset = dts_train, 
                              batch_size = conf['batch'],
                              pin_memory=True,
                              num_workers = 2,
                              # shuffle = True,
                              sampler = sampler)



    loader_test = DataLoader(dataset = dts_test, 
                              batch_size = conf['batch'],
                              pin_memory=True,
                              num_workers = 2,
                              shuffle = True)


    input_dim = X_train.shape[-1]

    model = UNADON(input_dim, 20, conf['res'], conf['attn_layers'], conf['attn_hidden_size'])
    print('Number of parameters',count_parameters(model))

    model.to(device)
    conf['device'] = device

    train(model, loader_train, loader_test, device, conf['output_name'])


if __name__ == '__main__':
    np.random.seed(413)
    torch.manual_seed(413)
    main()

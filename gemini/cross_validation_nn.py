import copy
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, train_test_split
from torch.utils.data import (DataLoader, RandomSampler, SubsetRandomSampler,
                              TensorDataset)
from tqdm import tqdm

from evaluate_performance import evaluate_performance

np.random.seed(1)
torch.manual_seed(1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Net1(nn.Module):

    def __init__(self, ndim, d, nlabel):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        # an affine operation: y = Wx + b
        self.d = d
        self.fc = []
        self.fc.append(nn.Linear(ndim, d[0]))

        for i in range(len(d)-1):
            self.fc.append(nn.Linear(d[i], d[i+1]))
        self.fc.append(nn.Linear(d[-1], nlabel))
        # self.drop = nn.Dropout(p=0.5)
        # self.bn1 = nn.BatchNorm1d(d1)
        # self.bn2 = nn.BatchNorm1d(d2)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        for i in range(len(self.d)):
            x = self.fc[i](x)
            x = F.relu(x)
        x = self.fc[-1](x)
        return x


class Net(nn.Module):

    def __init__(self, ndim, d, nlabel):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        # an affine operation: y = Wx + b

        self.d = d
        self.fc1 = nn.Linear(ndim, d[0])  # 5*5 from image dimension
        self.fc2 = nn.Linear(d[0], d[1])
        if len(d) == 2:
            self.fc3 = nn.Linear(d[-1], nlabel)
        elif len(d) == 3:
            self.fc3 = nn.Linear(d[1], d[2])
            self.fc4 = nn.Linear(d[-1], nlabel)

        self.drop = nn.Dropout(p=0.5)
        # self.bn1 = nn.BatchNorm1d(d1)
        # self.bn2 = nn.BatchNorm1d(d2)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        # x = self.drop(x)
        x = self.fc1(x)
        # x = self.bn1(x)
        x = F.relu(x)
        # x = self.drop(x)
        x = self.fc2(x)
        # x = self.bn2(x)
        x = F.relu(x)
        x = self.fc3(x)

        if len(self.d) == 3:
            x = F.relu(x)
            x = self.fc4(x)
        return x


def print_log(epoch, current_loss, val_loss,
              best_epoch, no_optim_epoch, best_val_loss, i=10):
    best_epoch = 0 if best_epoch is None else best_epoch
    print(
        f'Epoch {epoch+1}:')
    print(
        f'TrainLoss {current_loss/(i+1):.6f} ValLoss {val_loss:.6f}')
    print(
        f'BestEpoch: {best_epoch:3d} NoOptim {no_optim_epoch:2d}')
    print(best_val_loss)
    print()


def cross_validation_nn_pairs(x, nperm, batch_size,
                              ratio, target_pair_drugs, drug2embd, data3,
                              best_epoch_=None):
    # Scale features
    torch.manual_seed(1)
    maxval = np.expand_dims(np.max(x, axis=1), axis=1)
    minval = np.expand_dims(np.min(x, axis=1), axis=1)
    x = (x - minval) * (1 / (maxval - minval))
    x = x.T

    acc = np.zeros((nperm, 1))
    f1 = np.zeros((nperm, 1))
    aupr = np.zeros((nperm, 1))
    roc = np.zeros((nperm, 1))
    drugs = list(drug2embd.keys())

    nclass = 1
    num_epochs = 10000
    max_no_optim_epoch = 200
    net_arch = (200, 100)
    drugs = np.array(sorted(list(drug2embd.keys())))
    nperm = 5
    kf = KFold(n_splits=nperm, random_state=1, shuffle=True)
    # for fold, (test_drugs_ids, train_drugs_full_ids) in \
    for fold, (train_drugs_full_ids, test_drugs_ids) in \
            enumerate(kf.split(drugs)):
        # idxx = 0
        # for fold in range(nperm):
        #     kf = KFold(n_splits=3, random_state=fold, shuffle=True)
        #     for _, (test_drugs_ids, train_drugs_full_ids) in \
        #             enumerate(kf.split(drugs)):
        #         idxx += 1
        torch.manual_seed(1)
        train_drugs_full = drugs[train_drugs_full_ids]
        test_drugs = drugs[test_drugs_ids]
        train_drugs_ids, validation_drugs_ids = train_test_split(
            train_drugs_full_ids, test_size=ratio, random_state=1)
        train_drugs = drugs[train_drugs_ids]
        validation_drugs = drugs[validation_drugs_ids]
        X_train, Y_train = output_pair(
            target_pair_drugs, drug2embd, data3, train_drugs)
        X_train_full, Y_train_full = output_pair(
            target_pair_drugs, drug2embd, data3, train_drugs_full)
        X_validation, Y_validation = output_pair(
            target_pair_drugs, drug2embd, data3, validation_drugs)
        X_test, Y_test = output_pair(
            target_pair_drugs, drug2embd, data3, test_drugs)

        tensor_x_train = torch.Tensor(X_train).to(device)
        tensor_y_train = torch.Tensor(Y_train).to(device)
        dataset_train = TensorDataset(
            tensor_x_train, tensor_y_train)
        tensor_x_train_full = torch.Tensor(X_train_full).to(device)
        tensor_y_train_full = torch.Tensor(Y_train_full).to(device)
        dataset_train_full = TensorDataset(
            tensor_x_train_full, tensor_y_train_full)
        tensor_x_validation = torch.Tensor(X_validation).to(device)
        tensor_y_validation = torch.Tensor(Y_validation).to(device)
        dataset_validation = TensorDataset(
            tensor_x_validation, tensor_y_validation)
        tensor_x_test = torch.Tensor(X_test).to(device)
        tensor_y_test = torch.Tensor(Y_test).to(device)
        dataset_test = TensorDataset(tensor_x_test, tensor_y_test)

        train_subsampler = RandomSampler(dataset_train)
        train_subsampler_full = RandomSampler(dataset_train_full)
        validation_subsampler = RandomSampler(dataset_validation)
        test_subsampler = RandomSampler(dataset_test)

        g = torch.Generator()
        g.manual_seed(1)
        trainloader = DataLoader(
            dataset_train, generator=g,
            batch_size=batch_size, sampler=train_subsampler)
        trainloader_full = DataLoader(
            dataset_train_full, generator=g,
            batch_size=batch_size, sampler=train_subsampler_full)
        validationloader = DataLoader(
            dataset_validation, generator=g,
            batch_size=X_validation.shape[0], sampler=validation_subsampler)
        testloader = DataLoader(
            dataset_test, generator=g,
            batch_size=X_test.shape[0], sampler=test_subsampler)

        best_val_loss = np.inf
        no_optim_epoch = 0
        model = Net(X_train.shape[1], net_arch, nclass).to(device)
        loss_function = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters())
        # Print
        start_time = time.time()
        print(f'FOLD {fold}')
        print('--------------------------------')
        # best_epoch = 1
        best_epoch = best_epoch_
        if best_epoch is None:
            for epoch in range(num_epochs):
                # Set current loss value
                current_loss = 0.0
                # Iterate over the DataLoader for training data
                for i, data in enumerate(trainloader, 0):
                    # Get inputs
                    inputs, targets = data
                    # Zero the gradients
                    optimizer.zero_grad()
                    # Perform forward pass
                    outputs = model(inputs)

                    # Compute loss
                    loss = loss_function(outputs, targets)
                    # Perform backward pass
                    loss.backward()
                    # Perform optimization
                    optimizer.step()
                    # Print statistics
                    current_loss += loss.item()
                with torch.no_grad():
                    # Iterate over the test data and generate predictions
                    for idx, data in enumerate(validationloader, 0):
                        # Get inputs
                        inputs, targets = data
                        outputs = model(inputs)
                        val_loss = loss_function(outputs, targets)
                if val_loss < best_val_loss:
                    best_val_loss = float(val_loss.cpu().numpy().copy())
                    best_epoch = epoch
                    no_optim_epoch = 0
                else:
                    no_optim_epoch += 1
                if epoch % 10 == 9:
                    print_log(epoch, current_loss, val_loss,
                              best_epoch, no_optim_epoch, best_val_loss)
                if no_optim_epoch >= max_no_optim_epoch:
                    break

            print(
                f'Epoch {epoch+1}:')
            print(
                f'Train loss {current_loss/(i+1):.6f} Val loss {val_loss:.6f}')
            print(
                f'Best epoch: {best_epoch:3d} no_optim: {no_optim_epoch:2d}')

        model_full = Net(X_train.shape[1], net_arch, nclass).to(device)
        loss_function = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model_full.parameters())
        for epoch in tqdm(range(best_epoch)):
            # Set current loss value
            current_loss = 0.0
            # Iterate over the DataLoader for training data
            for i, data in enumerate(trainloader_full, 0):
                # Get inputs
                inputs, targets = data
                # Zero the gradients
                optimizer.zero_grad()
                # Perform forward pass
                outputs = model_full(inputs)

                # Compute loss
                loss = loss_function(outputs, targets)
                # Perform backward pass
                loss.backward()
                # Perform optimization
                optimizer.step()

            # Process is complete.

        # print('Training process has finished. Saving trained model.')
        print('Training process has finished.')
        # Print about testing
        print('Starting testing')
        # Saving the model
        # save_path = f'./model-fold-{fold}.pth'
        # torch.save(model.state_dict(), save_path)
        # Evaluationfor this fold
        with torch.no_grad():
            # Iterate over the test data and generate predictions
            for i, data in enumerate(testloader, 0):
                # Get inputs
                inputs, targets = data
                # Generate outputs
                # class_score = model(inputs)
                class_score_full = model_full(inputs)

        # class_score = class_score.cpu().numpy()
        class_score_full = class_score_full.cpu().numpy()
        label = targets.cpu().numpy()
        # Print accuracy
        # acc[fold], f1[fold], aupr[fold], roc[fold] = evaluate_performance(
        #     class_score, label, alpha=alpha)
        # print('[Trial #%d] acc: %f, f1: %f, auprc: %f, MAPRC: %f\n' %
        #       (fold+1, acc[fold], f1[fold], aupr[fold], roc[fold]))
        acc[fold], f1[fold], aupr[fold], roc[fold] = evaluate_performance(
            class_score_full, label, alpha=1)

        print('[Trial #%d] acc: %f, f1: %f, auprc: %f, MAPRC: %f\n' %
              (fold+1, acc[fold], f1[fold], aupr[fold], roc[fold]))
        end_time = time.time()
        print(f'Time: {end_time-start_time}\n')
    return acc, f1, aupr, roc


def cross_validation_nn(x, anno, nperm, batch_size=128,
                        alpha=3, ratio=0.2, best_epoch_=None,
                        task_type='classification', NN_stru=(200, 100),
                        train_test_ids=None, train_val_ids=None,
                        return_pred=False, max_no_optim_epoch=50,
                        train_on_full=False
                        ):
    # Scale features
    torch.manual_seed(1)
    maxval = np.expand_dims(np.max(x, axis=1), axis=1)
    minval = np.expand_dims(np.min(x, axis=1), axis=1)
    x = (x - minval) * (1 / (maxval - minval))

    tensor_x = torch.Tensor(x.T).to(device)  # transform to torch tensor
    tensor_y = torch.Tensor(anno.T).to(device)

    dataset = TensorDataset(tensor_x, tensor_y)  # create your datset

    (nclass, ngene) = np.shape(anno)

    acc = np.zeros((nperm, 1))
    f1 = np.zeros((nperm, 1))
    aupr = np.zeros((nperm, 1))
    roc = np.zeros((nperm, 1))
    spearmanr = np.zeros((nperm, 1))
    spearman_pvalue = np.zeros((nperm, 1))
    mse = np.zeros((nperm, 1))

    acc_train = np.zeros((nperm, 1))
    f1_train = np.zeros((nperm, 1))
    aupr_train = np.zeros((nperm, 1))
    roc_train = np.zeros((nperm, 1))
    spearmanr_train = np.zeros((nperm, 1))
    spearman_pvalue_train = np.zeros((nperm, 1))
    mse_train = np.zeros((nperm, 1))

    class_score_fulls = []
    labels = []
    test_ids_lst = []

    num_epochs = 100000
    if train_test_ids is None:
        kf = KFold(n_splits=nperm, random_state=1, shuffle=True)
        train_test_ids = kf.split(range(len(dataset)))
    for fold, (train_ids_full, test_ids) in enumerate(train_test_ids):
        best_epoch = best_epoch_
        torch.manual_seed(fold)
        if train_val_ids is not None:
            train_ids, validation_ids = train_val_ids[fold]
        else:
            train_ids, validation_ids = train_test_split(
                train_ids_full, test_size=ratio, random_state=1)
        model = Net(x.shape[0], NN_stru, nclass).to(device)
        if task_type == 'classification':
            loss_function = nn.BCEWithLogitsLoss()
        elif task_type == 'regression':
            loss_function = nn.MSELoss()

        optimizer = torch.optim.Adam(model.parameters())
        # Print
        start_time = time.time()
        print(f'FOLD {fold+1}')
        print('--------------------------------')

        train_subsampler = SubsetRandomSampler(train_ids)
        train_subsampler_full = SubsetRandomSampler(train_ids_full)

        g = torch.Generator()
        g.manual_seed(1)
        trainloader = DataLoader(
            dataset, generator=g,
            batch_size=batch_size, sampler=train_subsampler)
        trainloader_full2 = DataLoader(
            dataset, generator=g,
            batch_size=len(train_ids_full), sampler=train_subsampler_full)
        trainloader_full = trainloader_full2

        trainloader_full = DataLoader(
            dataset, generator=g,
            batch_size=batch_size, sampler=train_subsampler_full)

        validationloader = DataLoader(
            dataset, shuffle=False,
            batch_size=len(validation_ids), sampler=validation_ids)

        testloader = DataLoader(
            dataset, shuffle=False,
            batch_size=len(test_ids), sampler=test_ids)
        best_val_loss = np.inf
        no_optim_epoch = 0
        if best_epoch is None:
            for epoch in range(num_epochs):
                # Set current loss value
                current_loss = 0.0
                # Iterate over the DataLoader for training data
                for i, data in enumerate(trainloader, 0):
                    # Get inputs
                    inputs, targets = data
                    # Zero the gradients
                    optimizer.zero_grad()
                    # Perform forward pass
                    outputs = model(inputs)

                    # Compute loss
                    loss = loss_function(outputs, targets)
                    # Perform backward pass
                    loss.backward()
                    # Perform optimization
                    optimizer.step()
                    # Print statistics
                    current_loss += loss.item()
                with torch.no_grad():
                    # Iterate over the test data and generate predictions
                    for idx, data in enumerate(validationloader, 0):
                        # Get inputs
                        inputs, targets = data
                        outputs = model(inputs)

                if task_type == 'classification':
                    val_loss = float(loss_function(
                        outputs, targets).cpu().numpy().copy())
                elif task_type == 'regression':
                    spearman_corr, _ = stats.spearmanr(
                        outputs.cpu().numpy().copy(),
                        targets.cpu().numpy().copy())
                    val_loss = 1 - spearman_corr
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_epoch = epoch
                    no_optim_epoch = 0
                    model_best = copy.deepcopy(model)
                else:
                    no_optim_epoch += 1
                if epoch % 100 == 9:
                    print_log(epoch, current_loss, val_loss,
                              best_epoch, no_optim_epoch, best_val_loss)
                if no_optim_epoch >= max_no_optim_epoch:
                    break
            best_epoch = 0 if best_epoch is None else best_epoch
            print(
                f'Epoch {epoch+1}:')
            print(
                f'Train loss {current_loss/(i+1):.6f} Val loss {val_loss:.6f}')
            print(
                f'Best epoch: {best_epoch:3d} no_optim: {no_optim_epoch:2d}')

        if train_on_full is False:
            model_full = model_best
        else:
            torch.manual_seed(fold)
            if task_type == 'classification':
                loss_function = nn.BCEWithLogitsLoss()
            elif task_type == 'regression':
                loss_function = nn.MSELoss()
            model_full = Net(x.shape[0], NN_stru, nclass).to(device)
            optimizer = torch.optim.Adam(model_full.parameters())
            current_loss = 0.0
            for epoch in tqdm(range(best_epoch)):
                # Set current loss value
                current_loss = 0.0
                # Iterate over the DataLoader for training data
                for i, data in enumerate(trainloader_full, 0):
                    # Get inputs
                    inputs, targets = data
                    # Zero the gradients
                    optimizer.zero_grad()
                    # Perform forward pass
                    outputs = model_full(inputs)

                    # Compute loss
                    loss = loss_function(outputs, targets)
                    # Perform backward pass
                    loss.backward()
                    # Perform optimization
                    optimizer.step()
            # Process is complete.

        # print('Training process has finished. Saving trained model.')
        print('Training process has finished.')
        # Print about testing
        print('Starting testing')
        # Saving the model
        # save_path = f'./model-fold-{fold}.pth'
        # torch.save(model.state_dict(), save_path)

        # t = time.time()
        with torch.no_grad():
            # Iterate over the test data and generate predictions
            for i, data in enumerate(testloader, 0):
                # Get inputs
                inputs, targets = data
                # Generate outputs
                # class_score = model(inputs)
                class_score_full = model_full(inputs)
                loss = loss_function(class_score_full, targets)
        print(
            f'Test loss: {loss.item():.6f}')
        # print(class_score_full.shape)

        # class_score = class_score.cpu().numpy()
        class_score_full = class_score_full.cpu().numpy()
        label = targets.cpu().numpy()
        # Print accuracy

        # print(time.time()-t)

        if task_type == 'classification':
            acc[fold], f1[fold], aupr[fold], roc[fold] = evaluate_performance(
                class_score_full, label, alpha=alpha)
            print('[Trial #%d] acc: %f, f1: %f, auprc: %f, MAPRC: %f\n' %
                  (fold+1, acc[fold], f1[fold], aupr[fold], roc[fold]))
        elif task_type == 'regression':
            spearmanr[fold], spearman_pvalue[fold] = stats.spearmanr(
                class_score_full, label)
            mse[fold] = mean_squared_error(class_score_full, label)
            print('[Trial #%d] spearmanr: %f, spearman_pvalue: %f, mse: %f\n' %
                  (fold+1, spearmanr[fold], spearman_pvalue[fold], mse[fold]))
        # print(time.time()-t)

        class_score_fulls.append(class_score_full)
        labels.append(label)
        test_ids_lst.append(test_ids)

        end_time = time.time()
        print(f'Time: {end_time-start_time}\n')
    if task_type == 'classification':
        acc_train, f1_train, aupr_train, roc_train = \
            acc, f1, \
            aupr, roc
        if return_pred:
            return acc_train, f1_train, aupr_train, roc_train, acc, f1, \
                aupr, roc, class_score_fulls, labels, test_ids_lst
        else:
            return acc_train, f1_train, aupr_train, roc_train, acc, f1, \
                aupr, roc
    elif task_type == 'regression':
        spearmanr_train, spearman_pvalue_train, mse_train = \
            spearmanr, spearman_pvalue, mse
        if return_pred:
            return spearmanr_train, spearman_pvalue_train, mse_train, \
                spearmanr, spearman_pvalue, mse, class_score_fulls, \
                labels, test_ids_lst
        else:
            return spearmanr_train, spearman_pvalue_train, mse_train, \
                spearmanr, spearman_pvalue, mse


def validation_nn_output(x, anno, best_epoch_=100, batch_size=128,
                         task_type='classification', NN_stru=(200, 100),
                         ):
    # Scale features
    torch.manual_seed(1)
    maxval = np.expand_dims(np.max(x, axis=1), axis=1)
    minval = np.expand_dims(np.min(x, axis=1), axis=1)
    x = (x - minval) * (1 / (maxval - minval))

    tensor_x = torch.Tensor(x.T).to(device)  # transform to torch tensor
    tensor_y = torch.Tensor(anno.T).to(device)

    dataset = TensorDataset(tensor_x, tensor_y)  # create your datset

    (nclass, ngene) = np.shape(anno)

    train_ids_full = np.arange(x.shape[1])
    train_subsampler_full = SubsetRandomSampler(train_ids_full)

    g = torch.Generator()
    g.manual_seed(1)
    # trainloader_full2 = DataLoader(
    #     dataset, generator=g,
    #     batch_size=len(train_ids_full), sampler=train_subsampler_full)
    # trainloader_full = trainloader_full2

    trainloader_full = DataLoader(
        dataset, generator=g,
        batch_size=batch_size, sampler=train_subsampler_full)

    torch.manual_seed(1)
    if task_type == 'classification':
        loss_function = nn.BCEWithLogitsLoss()
    elif task_type == 'regression':
        loss_function = nn.MSELoss()
    model_full = Net(x.shape[0], NN_stru, nclass).to(device)
    optimizer = torch.optim.Adam(model_full.parameters())
    print(best_epoch_)
    for epoch in tqdm(range(best_epoch_)):
        # Set current loss value
        # Iterate over the DataLoader for training data
        for i, data in enumerate(trainloader_full, 0):
            # Get inputs
            inputs, targets = data
            # Zero the gradients
            optimizer.zero_grad()
            # Perform forward pass
            outputs = model_full(inputs)

            # Compute loss
            loss = loss_function(outputs, targets)
            # Perform backward pass
            loss.backward()
            # Perform optimization
            optimizer.step()
    # Process is complete.

    # # print('Training process has finished. Saving trained model.')
    # print('Training process has finished.')
    # # Print about testing
    # print('Starting testing')
    # # Saving the model
    # # save_path = f'./model-fold-{fold}.pth'
    # # torch.save(model.state_dict(), save_path)

    # t = time.time()
    class_scores = []
    with torch.no_grad():
        # Iterate over the test data and generate predictions
        for i, data in enumerate(trainloader_full, 0):
            # Get inputs
            inputs, targets = data
            # Generate outputs
            # class_score = model(inputs)
            class_score_full = model_full(inputs)
            class_score_full = class_score_full.cpu().numpy()
            class_scores.append(class_score_full)
    class_scores = np.concatenate(class_scores, axis=0)
    return class_scores

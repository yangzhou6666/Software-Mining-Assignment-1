'''This file contains several classifiers implemented using pytorch'''
import torch
from tqdm import tqdm
import numpy as np
import random
from torch.nn.functional import normalize
from sklearn.preprocessing import StandardScaler

class LogisticRegression(torch.nn.Module):
     def __init__(self, input_dim, output_dim):
         super(LogisticRegression, self).__init__()
         self.linear = torch.nn.Linear(input_dim, output_dim)
     def forward(self, x):
         outputs = torch.sigmoid(self.linear(x))
         return outputs

def load_data(train_path, test_path):
    '''load data from file'''
    with open(train_path) as f:
        train_lines = f.readlines()
        X_train = []
        Y_train = []
        for line in train_lines:
            line = line.split('#')[0].strip()
            line = line.split(' ')
            if line[0] == '+1':
                Y_train.append(1)
            else:
                Y_train.append(0)
            
            X_train.append([float(x.split(':')[-1]) for x in line[1:]])

    X_test = []
    Y_test = []
    with open(test_path) as f:
        test_lines = f.readlines()
        for line in test_lines:
            line = line.split('#')[0].strip()
            line = line.split(' ')
            if line[0] == '+1':
                Y_test.append(1)
            else:
                Y_test.append(0)
            X_test.append([float(x.split(':')[-1]) for x in line[1:]])

    # normalize data
    X_train = StandardScaler().fit_transform(X_train)
    X_test = StandardScaler().fit_transform(X_test)

    return X_train, Y_train, X_test, Y_test


def run_logistic_regression(train_path, test_path):
    # load data
    X_train, Y_train, X_test, Y_test = load_data(train_path, test_path)
    X_train, X_test = torch.Tensor(X_train),torch.Tensor(X_test)
    y_train, y_test = torch.Tensor(Y_train),torch.Tensor(Y_test)

    # create a logistic regression model
    epochs = 200
    input_dim = 50 # number of features
    output_dim = 1 # Single binary output 
    learning_rate = 0.01
    model = LogisticRegression(input_dim=input_dim, output_dim=output_dim)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    losses = []
    losses_test = []
    Iterations = []
    iter = 0
    for epoch in tqdm(range(int(epochs)),desc='Training Epochs'):
        x = X_train
        labels = y_train
        optimizer.zero_grad() # Setting our stored gradients equal to zero
        outputs = model(x)
        loss = criterion(torch.squeeze(outputs), labels) 
        
        loss.backward() # Computes the gradient of the given tensor w.r.t. the weights/bias
        optimizer.step() # Updates weights and biases with the optimizer (SGD)
        
        
        if iter%10==0:
            with torch.no_grad():
                # Calculating the loss and accuracy for the test dataset
                correct_test = 0
                total_test = 0
                outputs_test = torch.squeeze(model(X_test))
                loss_test = criterion(outputs_test, y_test)
                
                predicted_test = outputs_test.round().detach().numpy()
                total_test += y_test.size(0)
                correct_test += np.sum(predicted_test == y_test.detach().numpy())
                accuracy_test = 100 * correct_test/total_test
                losses_test.append(loss_test.item())
                
                # Calculating the loss and accuracy for the train dataset
                total = 0
                correct = 0
                total += y_train.size(0)
                correct += np.sum(torch.squeeze(outputs).round().detach().numpy() == y_train.detach().numpy())
                accuracy = 100 * correct/total
                losses.append(loss.item())
                Iterations.append(iter)
                
                print(f"Iteration: {iter}. \nTest - Loss: {loss_test.item()}. Accuracy: {accuracy_test}")
                print(f"Train -  Loss: {loss.item()}. Accuracy: {accuracy}\n")

        iter+=1


    

if __name__=="__main__":
    split_type = ['k_folder', 's_k_folder']
    train_path = './part_2/k_folder/train_1.txt'
    test_path = './part_2/k_folder/test_1.txt'
    # train_path = './part_1/train.txt'
    # test_path = './part_1/test.txt'

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    run_logistic_regression(train_path, test_path)
    

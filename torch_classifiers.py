'''This file contains several classifiers implemented using pytorch'''
import torch
from tqdm import tqdm
import numpy as np
import random
from torch.nn.functional import normalize
from sklearn.preprocessing import StandardScaler
import os 
from pykeops.torch import LazyTensor
from Sklearn_PyTorch import TorchRandomForestClassifier
import torch.nn.functional as F

np.seterr(invalid="ignore")

class LogisticRegression(torch.nn.Module):
     def __init__(self, input_dim, output_dim):
         super(LogisticRegression, self).__init__()
         self.linear = torch.nn.Linear(input_dim, output_dim)
     def forward(self, x):
         outputs = torch.sigmoid(self.linear(x))
         return outputs


class DNN(torch.nn.Module):
     def __init__(self, input_dim, output_dim):
         super(DNN, self).__init__()
         self.linear1 = torch.nn.Linear(input_dim, 50)
         self.linear2 = torch.nn.Linear(50, 20)
         self.linear3 = torch.nn.Linear(20, output_dim)
         
     def forward(self, x):
         x = torch.sigmoid(self.linear1(x))
         x = torch.sigmoid(self.linear2(x))
         outputs = torch.sigmoid(self.linear3(x))
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


def run_DNN(train_path, test_path, vbose=False):
    # load data
    X_train, Y_train, X_test, Y_test = load_data(train_path, test_path)
    X_train, X_test = torch.Tensor(X_train),torch.Tensor(X_test)
    y_train, y_test = torch.Tensor(Y_train),torch.Tensor(Y_test)

    # create a logistic regression model
    epochs = 2000
    input_dim = 50 # number of features
    output_dim = 1 # Single binary output 
    learning_rate = 0.01
    model = DNN(input_dim=input_dim, output_dim=output_dim)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    losses = []
    losses_test = []
    Iterations = []
    iter = 0
    for epoch in range(int(epochs)):
        x = X_train
        labels = y_train
        optimizer.zero_grad() # Setting our stored gradients equal to zero
        outputs = model(x)
        loss = criterion(torch.squeeze(outputs), labels) 
        
        loss.backward() # Computes the gradient of the given tensor w.r.t. the weights/bias
        optimizer.step() # Updates weights and biases with the optimizer (SGD)
        
        
        if epoch%10==0:
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

                TP = np.sum(predicted_test * y_test.detach().numpy())
                FP = np.sum(predicted_test * (1-y_test.detach().numpy()))
                FN = np.sum((1-predicted_test) * y_test.detach().numpy())
                precision = round(TP/(TP+FP) * 100, 2)
                recall = round(TP/(TP+FN) * 100, 2)
                f_1 = round(2*precision*recall/(precision+recall), 2)
                
                # Calculating the loss and accuracy for the train dataset
                total = 0
                correct = 0
                total += y_train.size(0)
                correct += np.sum(torch.squeeze(outputs).round().detach().numpy() == y_train.detach().numpy())
                accuracy = 100 * correct/total
                losses.append(loss.item())
                Iterations.append(iter)
                if vbose:
                    print(f"Iteration: {iter}. \nTest - Loss: {loss_test.item()}. Accuracy: {accuracy_test}. Precision: {precision}. Recall: {recall}. F1: {f_1}")
                    print(f"Train -  Loss: {loss.item()}. Accuracy: {accuracy}\n")


    return accuracy_test, precision, recall, f_1

def run_logistic_regression(train_path, test_path, vbose=False):
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
    for epoch in range(int(epochs)):
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

                TP = np.sum(predicted_test * y_test.detach().numpy())
                FP = np.sum(predicted_test * (1-y_test.detach().numpy()))
                FN = np.sum((1-predicted_test) * y_test.detach().numpy())
                precision = round(TP/(TP+FP) * 100, 2)
                recall = round(TP/(TP+FN) * 100, 2)
                f_1 = round(2*precision*recall/(precision+recall), 2)
                
                # Calculating the loss and accuracy for the train dataset
                total = 0
                correct = 0
                total += y_train.size(0)
                correct += np.sum(torch.squeeze(outputs).round().detach().numpy() == y_train.detach().numpy())
                accuracy = 100 * correct/total
                losses.append(loss.item())
                Iterations.append(iter)
                if vbose:
                    print(f"Iteration: {iter}. \nTest - Loss: {loss_test.item()}. Accuracy: {accuracy_test}. Precision: {precision}. Recall: {recall}. F1: {f_1}")
                    print(f"Train -  Loss: {loss.item()}. Accuracy: {accuracy}\n")

        iter+=1

    return accuracy_test, precision, recall, f_1




def run_random_forest(train_path, test_path, vbose=False):
    # load data
    X_train, Y_train, X_test, Y_test = load_data(train_path, test_path)
    X_train, X_test = torch.Tensor(X_train),torch.Tensor(X_test)
    y_train, y_test = torch.Tensor(Y_train),torch.Tensor(Y_test)
    my_model = TorchRandomForestClassifier(nb_trees=10, nb_samples=X_train.shape[0], max_depth=3, bootstrap=True)
    my_model.fit(X_train, y_train)
    
    # Prediction function
    predicted_results = []
    for input in X_test:
        predicted_results.append(my_model.predict(input))

    accuracy = 100 * np.sum(predicted_results == y_test.detach().numpy()) / len(predicted_results)
    
    TP = np.sum(predicted_results * y_test.detach().numpy())
    FP = np.sum(predicted_results * (1-y_test.detach().numpy()))
    FN = np.sum((1-np.array(predicted_results)) * y_test.detach().numpy())
    precision = round(TP/(TP+FP) * 100, 2)
    recall = round(TP/(TP+FN) * 100, 2)
    f_1 = round(2*precision*recall/(precision+recall), 2)
    if vbose:
        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1: {f_1}")
    return accuracy, precision, recall, f_1


def main_lr(split_type, root_data, k):
    for type in split_type:
        accs = []
        precisions = []
        recalls = []
        f1s = []
        for i in range(k):
            train_path = os.path.join(root_data, type, f'train_{i}.txt')
            test_path = os.path.join(root_data, type, f'test_{i}.txt')
            accuracy_test, precision, recall, f_1 = run_logistic_regression(train_path, test_path)
            accs.append(accuracy_test)
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f_1)
        
        print(f"Results for {type} split:")
        for i in range(k):
            print('|' + str(i) + '|' + str(accs[i]) + '%|' + str(precisions[i]) + '%|' + str(recalls[i]) + '%|' + str(f1s[i]) + '%|')
        
        print('|' + 'avg' + '|' + str(round(sum(accs) / k, 2)) + '%|' + str(round(sum(precisions) / k, 2)) + '%|' + str(round(sum(recalls) / k, 2)) + '%|' + str(round(sum(f1s) / k, 2)) + '%|')
        
        print("\n\n")

def main_dnn(split_type, root_data, k):
    for type in split_type:
        accs = []
        precisions = []
        recalls = []
        f1s = []
        print(f"Results for {type} split:")
        for i in range(k):
            train_path = os.path.join(root_data, type, f'train_{i}.txt')
            test_path = os.path.join(root_data, type, f'test_{i}.txt')
            accuracy_test, precision, recall, f_1 = run_DNN(train_path, test_path, False)
            accs.append(accuracy_test)
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f_1)
        
        
            print('|' + str(i) + '|' + str(accs[i]) + '%|' + str(precisions[i]) + '%|' + str(recalls[i]) + '%|' + str(f1s[i]) + '%|')
            # exit()
        
        print('|' + 'avg' + '|' + str(round(sum(accs) / k, 2)) + '%|' + str(round(sum(precisions) / k, 2)) + '%|' + str(round(sum(recalls) / k, 2)) + '%|' + str(round(sum(f1s) / k, 2)) + '%|')
        
        print("\n\n")



def main_rf(split_type, root_data, k):
    '''main function for random forest'''
    for type in split_type:
        accs = []
        precisions = []
        recalls = []
        f1s = []
        print(f"Results for {type} split:")
        for i in range(k):
            train_path = os.path.join(root_data, type, f'train_{i}.txt')
            test_path = os.path.join(root_data, type, f'test_{i}.txt')
            accuracy_test, precision, recall, f_1 = run_random_forest(train_path, test_path)
            accs.append(accuracy_test)
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f_1)
            print('|' + str(i) + '|' + str(accs[i]) + '%|' + str(precisions[i]) + '%|' + str(recalls[i]) + '%|' + str(f1s[i]) + '%|')
        
    
        
        print('|' + 'avg' + '|' + str(round(sum(accs) / k, 2)) + '%|' + str(round(sum(precisions) / k, 2)) + '%|' + str(round(sum(recalls) / k, 2)) + '%|' + str(round(sum(f1s) / k, 2)) + '%|')
        
        print("\n\n")


if __name__=="__main__":
    split_type = ['k_folder', 's_k_folder']
    root_data = './part_2'
    train_path = './part_2/k_folder/train_1.txt'
    test_path = './part_2/k_folder/test_1.txt'
    k = 10
    # train_path = './part_1/train.txt'
    # test_path = './part_1/test.txt'
    dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    main_dnn(split_type, root_data, k)
    main_lr(split_type, root_data, k)

            

    

'''functions to split datasets'''
import os
import random

def split_train_test(file_path, train_path, test_path, ratio=0.5):
    '''split dataset into train and test'''
    with open(file_path) as f:
        lines = f.readlines()
    random.shuffle(lines)
    train_lines = lines[:int(len(lines) * ratio)]
    test_lines = lines[int(len(lines) * ratio):]
    with open(train_path, 'w') as f:
        f.writelines(train_lines)
    with open(test_path, 'w') as f:
        f.writelines(test_lines)
    


if __name__=='__main__':
    random.seed(123456)
    file_path = './Description/resources/datafile.txt'
    train_path = './part_1/train.txt'
    test_path = './part_1/test.txt'
    os.makedirs(os.path.dirname(train_path), exist_ok=True)
    split_train_test(file_path, train_path, test_path)

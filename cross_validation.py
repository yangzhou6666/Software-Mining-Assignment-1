'''The file is for k-folder validation'''
import random
import os

def k_folder_split(file_path, save_path, k=10):
    '''split dataset into k-folders'''
    with open(file_path) as f:
        lines = f.readlines()
    random.shuffle(lines)
    size = len(lines) / k
    for i in range(k):
        train_path = os.path.join(save_path, 'train_' + str(i) + '.txt')
        test_path = os.path.join(save_path, 'test_' + str(i) + '.txt')
        with open(train_path, 'w') as f:
            f.writelines(lines[:int(size * i)])
            f.writelines(lines[int(size * (i + 1)):])
        with open(test_path, 'w') as f:
            f.writelines(lines[int(size * i):int(size * (i + 1))])
        
def stratified_k_folder_split(file_path, save_path, k=10):
    '''split dataset into stratified k-folders'''
    with open(file_path) as f:
        lines = f.readlines()
    random.shuffle(lines)

    # get positive and negative lines
    pos_lines = []
    neg_lines = []
    for line in lines:
        if line.split(' ')[0] == '1':
            pos_lines.append(line)
        else:
            neg_lines.append(line)

    # get the ratio of positive and negative lines
    pos_ratio = len(pos_lines) / len(lines)
    neg_ratio = len(neg_lines) / len(lines)

    # split the dataset into k-folders
    size = len(lines) / k
    for i in range(k):
        train_path = os.path.join(save_path, 'train_' + str(i) + '.txt')
        test_path = os.path.join(save_path, 'test_' + str(i) + '.txt')
        with open(train_path, 'w') as f:
            f.writelines(pos_lines[:int(size * pos_ratio * i)])
            f.writelines(neg_lines[:int(size * neg_ratio * i)])
            f.writelines(pos_lines[int(size * pos_ratio * (i + 1)):])
            f.writelines(neg_lines[int(size * neg_ratio * (i + 1)):])
        with open(test_path, 'w') as f:
            f.writelines(pos_lines[int(size * pos_ratio * i):int(size * pos_ratio * (i + 1))])
            f.writelines(neg_lines[int(size * neg_ratio * i):int(size * neg_ratio * (i + 1))])

if __name__=="__main__":
    random.seed(123456)
    file_path = './Description/resources/datafile.txt'
    save_path = './part_2/k_folder/'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    k_folder_split(file_path, save_path)

    save_path = './part_2/s_k_folder/'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    stratified_k_folder_split(file_path, save_path)
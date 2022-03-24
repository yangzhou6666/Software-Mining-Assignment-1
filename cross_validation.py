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
        


if __name__=="__main__":
    random.seed(123456)
    file_path = './Description/resources/datafile.txt'
    save_path = './part_2/'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    k_folder_split(file_path, save_path)
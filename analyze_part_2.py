'''scripts to analzye the part 2 results'''
import os

def analyze_svm(result_path, k=10):
    '''analyze the result of svm'''
    total_acc = []
    total_pre = []
    total_recall = []
    total_f_1 = []
    for i in range(k):
        with open(os.path.join(result_path, 'log_' + str(i) + '.txt')) as f:
            lines = f.readlines()
            acc = lines[-2].split('%')[0].split(' ')[-1]
            pre_recall_data = lines[-1].split(' ')[-1]
            pre = pre_recall_data.split('/')[0][0:-1]
            recall = pre_recall_data.split('/')[1][0:-2]
            f_1 = 2 * float(pre) * float(recall) / (float(pre) + float(recall))
            f_1 = round(f_1, 2)
            total_acc.append(float(acc))
            total_pre.append(float(pre))
            total_recall.append(float(recall))
            total_f_1.append(f_1)

            print('|' + str(i) + '|' + acc + '%|' + pre + '%|' + recall + '%|' + str(f_1) + '%|' )

    print('|' + 'total' + '|' + str(round(sum(total_acc) / k, 2)) + '%|' + str(round(sum(total_pre) / k, 2)) + '%|' + str(round(sum(total_recall) / k, 2)) + '%|' + str(round(sum(total_f_1) / k, 2)) + '%|')




if __name__ == '__main__':
    result_path = './part_2/'
    analyze_svm(result_path)
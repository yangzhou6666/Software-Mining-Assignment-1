
# Part 1

First, we split the original dataset into train and test sets.
```
python split.py
```

Before training the SVM classifier, we need to compile the tool.
```
cd svm_light
make
```

Then, we can train the SVM classifier on the training data.
```
svm_light/svm_learn part_1/train.txt part_1/model
```

Now we can use the obtained model to classify the test set.
```
svm_light/svm_classify part_1/test.txt part_1/model part_1/result.txt
```

We obtain the following results on the testing data:
* Accuracy on test set: 71.00% (71 correct, 29 incorrect, 100 total)
* Precision/recall on test set: 66.27%/98.21%
* $F_1 = \frac{2 * PR}{P + R} = 79.13\%$


# Part 2

First, we split the original dataset using 10-folder and stratified 10-folder cross validation. 
```
python cross_validation.py
```

The split results will be stored in `./part_2/k_folder` and `./part_2/s_k_folder`.



## SVM

Assuming that you have compiled SVM classifier in part 1, let's execute the following scripts to run SVM on cross-validation sets.
```
bash part_2_svm.sh
```

The results will be stored under `./part_2/k_folder/log_k.txt` and `./part_2/s_k_folder/log_k.txt`. We summarize the results in the table below.

### Results for 10-folder cross-validation

|  k   | Acc  | Prec | Reca | $F_1$ |
|  ----  | ----  |---- |---- |---- | 
|0|50.00%|33.33%|100.00%|50.0%|
|1|65.00%|57.14%|88.89%|69.56%|
|2|55.00%|45.45%|62.50%|52.63%|
|3|70.00%|68.75%|91.67%|78.57%|
|4|85.00%|80.00%|100.00%|88.89%|
|5|80.00%|71.43%|100.00%|83.33%|
|6|75.00%|70.59%|100.00%|82.76%|
|7|70.00%|62.50%|100.00%|76.92%|
|8|60.00%|55.56%|100.00%|71.43%|
|9|70.00%|72.22%|92.86%|81.25%|
|avg|68.0%|61.7%|93.59%|73.53%|

### results for stratified 10-folder cross-validation

|  k   | Acc  | Prec | Reca | $F_1$ |
|  ----  | ----  |---- |---- |---- | 
|0|85.00%|78.57%|100.00%|88.0%|
|1|40.00%|25.00%|100.00%|40.0%|
|2|75.00%|75.00%|92.31%|82.76%|
|3|65.00%|58.82%|100.00%|74.07%|
|4|65.00%|56.25%|100.00%|72.0%|
|5|75.00%|76.92%|83.33%|80.0%|
|6|70.00%|64.71%|100.00%|78.57%|
|7|70.00%|57.14%|100.00%|72.72%|
|8|75.00%|78.57%|84.62%|81.48%|
|9|60.00%|58.82%|90.91%|71.43%|
|avg|68.0%|62.98%|95.12%|74.1%|


## Logistic regression

For other models, you need to run
```
python torch_classifier.py
```

* Note!: it may throw a warning (that does not affect the running of the code) saying a header file can not be found. That is related to a dependency library. Just ignore the warning. 

### Results for 10-folder cross-validation

|  k   | Acc  | Prec | Reca | $F_1$ |
|  ----  | ----  |---- |---- |---- | 
|0|60.0%|38.46%|100.0%|55.55%|
|1|55.0%|50.0%|66.67%|57.14%|
|2|60.0%|50.0%|62.5%|55.56%|
|3|55.0%|61.54%|66.67%|64.0%|
|4|80.0%|83.33%|83.33%|83.33%|
|5|75.0%|69.23%|90.0%|78.26%|
|6|60.0%|64.29%|75.0%|69.23%|
|7|50.0%|50.0%|60.0%|54.55%|
|8|55.0%|54.55%|60.0%|57.15%|
|9|50.0%|66.67%|57.14%|61.54%|
|avg|60.0%|58.81%|72.13%|63.63%|

### results for stratified 10-folder cross-validation

|  k   | Acc  | Prec | Reca | $F_1$ |
|  ----  | ----  |---- |---- |---- | 
|0|80.0%|76.92%|90.91%|83.33%|
|1|35.0%|20.0%|75.0%|31.58%|
|2|60.0%|72.73%|61.54%|66.67%|
|3|65.0%|63.64%|70.0%|66.67%|
|4|50.0%|45.45%|55.56%|50.0%|
|5|85.0%|90.91%|83.33%|86.96%|
|6|80.0%|76.92%|90.91%|83.33%|
|7|55.0%|46.15%|75.0%|57.14%|
|8|65.0%|75.0%|69.23%|72.0%|
|9|45.0%|50.0%|45.45%|47.62%|
|avg|62.0%|61.77%|71.69%|64.53%|


## Random Forest

### Results for 10-folder cross-validation

|  k   | Acc  | Prec | Reca | $F_1$ |
|  ----  | ----  |---- |---- |---- | 
|0|70.0%|33.33%|20.0%|25.0%|
|1|65.0%|75.0%|33.33%|46.15%|
|2|45.0%|38.46%|62.5%|47.62%|
|3|70.0%|68.75%|91.67%|78.57%|
|4|85.0%|80.0%|100.0%|88.89%|
|5|60.0%|66.67%|40.0%|50.0%|
|6|80.0%|75.0%|100.0%|85.71%|
|7|50.0%|50.0%|30.0%|37.5%|
|8|70.0%|66.67%|80.0%|72.73%|
|9|55.0%|69.23%|64.29%|66.67%|
|avg|65.0%|62.31%|62.18%|59.88%|

### results for stratified 10-folder cross-validation

|  k   | Acc  | Prec | Reca | $F_1$ |
|  ----  | ----  |---- |---- |---- | 
|0|55.0%|75.0%|27.27%|40.0%|
|1|35.0%|20.0%|75.0%|31.58%|
|2|45.0%|75.0%|23.08%|35.3%|
|4|50.0%|33.33%|11.11%|16.66%|
|5|75.0%|76.92%|83.33%|80.0%|
|6|75.0%|68.75%|100.0%|81.48%|
|7|60.0%|50.0%|50.0%|50.0%|
|8|75.0%|75.0%|92.31%|82.76%|
|9|55.0%|57.14%|72.73%|64.0%|
|avg|57.5%|59.01%|53.48%|53.54%|

## Deep Neural Networks

### Results for 10-folder cross-validation

|  k   | Acc  | Prec | Reca | $F_1$ |
|  ----  | ----  |---- |---- |---- | 
|0|50.0%|30.77%|80.0%|44.45%|
|1|60.0%|53.85%|77.78%|63.64%|
|2|60.0%|50.0%|62.5%|55.56%|
|3|60.0%|64.29%|75.0%|69.23%|
|4|80.0%|83.33%|83.33%|83.33%|
|5|75.0%|66.67%|100.0%|80.0%|
|6|75.0%|81.82%|75.0%|78.26%|
|7|60.0%|58.33%|70.0%|63.63%|
|8|55.0%|55.56%|50.0%|52.63%|
|9|50.0%|66.67%|57.14%|61.54%|
|avg|62.5%|61.13%|73.07%|65.23%|

### results for stratified 10-folder cross-validation

|  k   | Acc  | Prec | Reca | $F_1$ |
|  ----  | ----  |---- |---- |---- | 
|0|85.0%|83.33%|90.91%|86.96%|
|1|35.0%|20.0%|75.0%|31.58%|
|2|65.0%|75.0%|69.23%|72.0%|
|3|70.0%|66.67%|80.0%|72.73%|
|4|50.0%|45.45%|55.56%|50.0%|
|5|80.0%|83.33%|83.33%|83.33%|
|6|75.0%|71.43%|90.91%|80.0%|
|7|55.0%|46.15%|75.0%|57.14%|
|8|65.0%|80.0%|61.54%|69.57%|
|9|55.0%|60.0%|54.55%|57.15%|
|avg|63.5%|63.14%|73.6%|66.05%|
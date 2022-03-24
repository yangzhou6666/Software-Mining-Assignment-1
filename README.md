
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

### Results for 10-folder cross-validation

|  k   | Acc  | Prec | Reca | $F_1$ |
|  ----  | ----  |---- |---- |---- | 
|0|45.0%|31.25%|100.0%|47.62%|
|1|75.0%|64.29%|100.0%|78.26%|
|2|60.0%|50.0%|62.5%|55.56%|
|3|45.0%|55.56%|41.67%|47.62%|
|4|70.0%|87.5%|58.33%|70.0%|
|5|55.0%|57.14%|40.0%|47.06%|
|6|50.0%|60.0%|50.0%|54.55%|
|7|50.0%|50.0%|50.0%|50.0%|
|8|55.0%|57.14%|40.0%|47.06%|
|9|55.0%|72.73%|57.14%|64.0%|
|avg|56.0%|58.56%|59.96%|56.17%|

### results for stratified 10-folder cross-validation

|  k   | Acc  | Prec | Reca | $F_1$ |
|  ----  | ----  |---- |---- |---- | 
|0|35.0%|25.0%|9.09%|13.33%|
|1|65.0%|28.57%|50.0%|36.36%|
|2|60.0%|77.78%|53.85%|63.64%|
|3|65.0%|66.67%|60.0%|63.16%|
|4|45.0%|41.67%|55.56%|47.62%|
|5|40.0%|50.0%|58.33%|53.84%|
|6|45.0%|50.0%|36.36%|42.1%|
|7|20.0%|21.43%|37.5%|27.27%|
|8|50.0%|63.64%|53.85%|58.34%|
|9|30.0%|36.36%|36.36%|36.36%|
|avg|45.5%|46.11%|45.09%|44.2%|

# Software-Mining-Assignment-1


## Part 1

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


## Part 2

First, we split the original dataset using 10-folder and stratified 10-folder cross validation. 
```
python cross_validation.py
```

The split results will be stored in `./part_2/k_folder` and `./part_2/s_k_folder`.

Assuming that you have compiled SVM classifier in part 1, let's execute the following scripts to run SVM on cross-validation sets.
```
bash part_2_svm.sh
```

The results will be stored under `./part_2/k_folder/log_k.txt` and `./part_2/s_k_folder/log_k.txt`. We summarize the results in the table below.

### Results for 10-folder cross-validation (SVM)

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

### results for stratified 10-folder cross-validation (SVM)

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
|total|68.0%|62.98%|95.12%|74.1%|


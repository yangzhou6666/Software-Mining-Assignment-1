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
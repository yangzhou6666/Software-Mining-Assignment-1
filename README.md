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
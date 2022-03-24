# 10-folder cross validation

for k in {0..9}
do
    svm_light/svm_learn part_2/k_folder/train_$k.txt part_2/k_folder/model_$k > part_2/k_folder/log_$k.txt 
done

for k in {0..9}
do  
    svm_light/svm_classify part_2/k_folder/test_$k.txt part_2/k_folder/model_$k part_2/k_folder/result_$k.txt >> part_2/k_folder/log_$k.txt 
done


# stratified 10-folder cross validation


for k in {0..9}
do
    svm_light/svm_learn part_2/s_k_folder/train_$k.txt part_2/s_k_folder/model_$k > part_2/s_k_folder/log_$k.txt 
done

for k in {0..9}
do  
    svm_light/svm_classify part_2/s_k_folder/test_$k.txt part_2/s_k_folder/model_$k part_2/s_k_folder/result_$k.txt >> part_2/s_k_folder/log_$k.txt 
done


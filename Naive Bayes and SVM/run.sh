#!/bin/bash
# $1 is 1st argument
if [ $1 -eq 1 ] 
then
	if [ $2 -eq 1 ]
	then
		python Q1/model1/A2_1_training.py $3 $4
	elif [ $2 -eq 2 ]
	then
		python Q1/model2/A2_1_training_st.py $3 $4
	elif [ $2 -eq 3 ]
	then
		python Q1/model3/A2_1_training_fe.py $3 $4
	else
		echo "Unknown option"
	fi
elif [ $1 -eq 2 ]
then
	if [ $2 -eq 1 ]
	then
		python Q2/model1/A2_2_testing.py $3 $4
	elif [ $2 -eq 2 ]
	then
		python Q2/model2/convert.py $3
		./svm-predict libsvm_test.txt libsvm_train.txt.model $4
	elif [ $2 -eq 3 ]
	then
		python Q2/model3/convert.py $3
		./svm-predict libsvm_test.txt libsvm_train_gaussian10.txt.model $4
	else
		echo "Unknown option"
	fi
else 
	echo "Unknown option"
fi

# ./run.sh <Question_number> <model_number> <input_file_name> <output_file_name>

#             Where

# <Question_number> = 1 for NB and

# <Question_number> = 2 for SVM and

# <model_number>  = 1, 2, or 3 for different parts, as defined later

# <input_file_name> = path to the input file containing test data. We have provided sample input files for both questions.

# <output_file_name> = path to the output file which should contain predictions for the samples in input file - one prediction in each row.

# <model_number> for Q1:

#      1 NB model corresponding to part-a i.e. without stemming and stopword removal.

#      2 NB model corresponding to part-d i.e. with stemming and stopword removal.

#      3 NB model corresponding to part-e, i.e. your best model.
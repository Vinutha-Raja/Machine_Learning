python3 process_training_data.py "car" "train.csv" "test.csv" "entropy" 6 "False"
python3 process_training_data.py "car" "train.csv" "test.csv" "majority_error" 6 "False"
python3 process_training_data.py "car" "train.csv" "test.csv" "gini_index" 6 "False"

python3 process_training_data.py "bank" "bank/train.csv" "bank/test.csv" "entropy" 16 "False"
python3 process_training_data.py "bank" "bank/train.csv" "bank/test.csv" "majority_error" 16 "False"
python3 process_training_data.py "bank" "bank/train.csv" "bank/test.csv" "gini_index" 16 "False"

python3 process_training_data.py "bank" "bank/train.csv" "bank/test.csv" "entropy" 16 "True"
python3 process_training_data.py "bank" "bank/train.csv" "bank/test.csv" "majority_error" 16 "True"
python3 process_training_data.py "bank" "bank/train.csv" "bank/test.csv" "gini_index" 16 "True"

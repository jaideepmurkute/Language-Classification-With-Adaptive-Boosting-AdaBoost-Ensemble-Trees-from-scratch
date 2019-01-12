# Language-Classification-With-Adaptive-Boosting-AdaBoost-Ensemble-Trees-from-scratch

**How to train model:**
python    train.py    mode='dt'or'ab'   train_file_name

sample:
python train.py ab twenty_words_sample.dat

training file should have following format:
nl|a sentence in Dutch.
en|a sentence in English.
(can refer to any train file like twenty_words_sample.dat in submitted folder.)

--------------------------------------------------------

**How to make predictions:**
python predict.py mode=ab|dt

sample: python predict.py dt new_nl.txt

test file should have a single plaintext data which we want to classify.

Note: predict.py file is modified to match up with the requirements of the assignments and will only use one best model. Code with minor changes can support different models.
---------------------------------------------------------

***Train Files:
ten_words_sample.dat
twenty_words_sample.dat
fifty_words_sample.dat

***Test Files:
ten_words_test.dat
twenty_words_test.dat
fifty_words_test.dat

----------------------------------------------------

**Being used by test file by default**
Favorite Model:
AdaBoostWeightsDump_fav

other trained models:
AdaBoostWeightsDump
DTree

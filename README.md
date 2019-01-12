# Language-Classification-With-Adaptive-Boosting-AdaBoost-Ensemble-Trees-from-scratch

**About**
Implementation for language classifier between English and Dutch languages. One model uses simple full grown decision tree model while another is the ensemble learner with adaptive boosting(AdaBoost).

Performance of both models is being validation on varying length input sizes of text and performance evaluation has been consolidated in results.pdf file.

----------------------------------------------------------

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

---------------------------------------------------------

**Train Files:**
ten_words_sample.dat
twenty_words_sample.dat
fifty_words_sample.dat

**Test Files:**
ten_words_test.dat
twenty_words_test.dat
fifty_words_test.dat

----------------------------------------------------

**Being used by test file by default:**

Favorite Model:
AdaBoostWeightsDump_fav

other trained models:
AdaBoostWeightsDump
DTree

import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score, roc_curve, auc

K = 500
TRIAL = 5

HD_TRAIN_FILE = 'split{}/hd_train.txt'
NHD_TRAIN_FILE = 'split{}/nhd_train.txt'

HD_TEST_FILE = 'split{}/hd_test.txt'
NHD_TEST_FILE = 'split{}/nhd_test.txt'

PATTERN_GASP_5 = 'split{}/TrainGASP5M.txt'
PATTERN_GASP_10 = 'split{}/TrainGASP10M.txt'
PATTERN_SPADE = 'split{}/seq_train_CMSPADE.txt'
PATTERN_SPAM = 'split{}/seq_train_CMSPAM.txt'
PATTERN_FIXED = 'split{}/GraSeq_fixed_5M.txt'
PATTERN_VAR = 'split{}/GraSeq_variable_5M.txt'

def read_data(hd_file, nhd_file):
	label = list()
	data = list()

	f = open(hd_file, 'r')
	for line in f:
		row = line.strip().split('-1')
		holder = list()
		for ccs_lst in row:
			if '-2' in ccs_lst:
				break
			temp = set(map(int, ccs_lst.split()))
			holder.append(temp)

		data.append(holder)
		label.append(1)
	f.close()

	f = open(nhd_file, 'r')
	for line in f:
		row = line.strip().split('-1')
		holder = list()
		for ccs_lst in row:
			if '-2' in ccs_lst:
				break
			temp = set(map(int, ccs_lst.split()))
			holder.append(temp)

		data.append(holder)
		label.append(0)
	f.close()

	return np.asarray(label), data


def read_pattern(filename):
	pattern = list()
	pattern_temp = list()

	f = open(filename, 'r')
	for line in f:
		temp = list()
		check = 0
		row = line.strip().split('-1')
		for ccs_lst in row:
			if '-2' in ccs_lst or 'SUP' in ccs_lst:
				cnt = int(ccs_lst.split()[-1])
			else:
				check += 1
				temp.append(set(map(int, ccs_lst.split())))
		if check == 1:
			continue
		pattern_temp.append([temp, cnt])
	f.close()

	sorted_lst = sorted(pattern_temp, key=lambda x: x[1], reverse=True)

	for p in sorted_lst[:K]:
		pattern.append(p[0])

	return pattern


def set_feature(x, pattern):
	feature = list()

	for d in x:
		temp = [0 for _ in range(len(pattern))]
		for i in range(len(pattern)):
			if len(pattern[i]) > len(d):
				continue
			p = pattern[i]

			pi = 0
			for di in d:
				if pi == len(p):
					break

				if p[pi].issubset(di):
					pi += 1

			if pi == len(p):
				temp[i] = 1
		feature.append(temp)

	return np.asarray(feature)


def learn_model(y_train, x_train, y_test, x_test):
	clf = xgb.XGBClassifier(learning_rate = 0.05, n_estimators=100, max_depth=3)
	y_pred = clf.fit(x_train, y_train).predict_proba(x_test)[:,1]

	fpr, tpr, _ = roc_curve(y_test, y_pred)
	auc_score = auc(fpr, tpr)

	print('\t',auc_score)

	return fpr, tpr, auc_score


def run(filename, x_train_data, y_train, x_test_data, y_test):
	pattern = read_pattern(filename)
	x_train = set_feature(x_train_data, pattern)
	x_test = set_feature(x_test_data, pattern)
	fpr, tpr, auc_score = learn_model(y_train, x_train, y_test, x_test)
	return fpr, tpr, auc_score

if __name__ == '__main__':
	auc_5 = 0.0
	auc_10 = 0.0
	auc_spade = 0.0
	auc_spam = 0.0
	auc_fixed = 0.0
	auc_var = 0.0
	for trial_num in range(1, TRIAL+1):
		hd_train = HD_TRAIN_FILE.format(trial_num)
		nhd_train = NHD_TRAIN_FILE.format(trial_num)
		hd_test = HD_TEST_FILE.format(trial_num)
		nhd_test = NHD_TEST_FILE.format(trial_num)

		y_train, x_train_data = read_data(hd_train, nhd_train)
		y_test, x_test_data = read_data(hd_test, nhd_test)

		pattern_file = PATTERN_GASP_5.format(trial_num)
		fpr, tpr, auc_score = run(pattern_file, x_train_data, y_train, x_test_data, y_test)
		auc_5 += auc_score

		pattern_file = PATTERN_GASP_10.format(trial_num)
		fpr, tpr, auc_score = run(pattern_file, x_train_data, y_train, x_test_data, y_test)
		auc_10 += auc_score

		pattern_file = PATTERN_SPADE.format(trial_num)
		fpr, tpr, auc_score = run(pattern_file, x_train_data, y_train, x_test_data, y_test)
		auc_spade += auc_score

		pattern_file = PATTERN_SPAM.format(trial_num)
		fpr, tpr, auc_score = run(pattern_file, x_train_data, y_train, x_test_data, y_test)
		auc_spam += auc_score

		pattern_file = PATTERN_FIXED.format(trial_num)
		fpr, tpr, auc_score = run(pattern_file, x_train_data, y_train, x_test_data, y_test)
		auc_fixed += auc_score

		pattern_file = PATTERN_VAR.format(trial_num)
		fpr, tpr, auc_score = run(pattern_file, x_train_data, y_train, x_test_data, y_test)
		auc_var += auc_score

	print('GASP-5M:', auc_5/TRIAL)
	print('GASP-10M:', auc_10/TRIAL)
	print('CM-SPADE:', auc_spade/TRIAL)
	print('CM-SPAM:', auc_spam/TRIAL)
	print('GraSeq-Fixed:', auc_var/TRIAL)
	print('GraSeq-Var:', auc_var/TRIAL)




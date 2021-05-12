import os
import datetime
import numpy as np

# N_data.txt: patient_id date ccs1 ccs2 ...
# N_label.txt: patient_id label
PP_PATH = 'dataset/pp_file/'
SAMPLE_PATH = 'dataset/sample_file/'

DAY_DELTA = 7
MIN_TRAIN_SIZE = 500
MAX_TRAIN_SIZE = 10000
TEST_SIZE = 2000

np.random.seed(1001)

class ReorganizeData():
	def __init__(self, day_delta, min_train_szie, max_train_size, test_size):
		self.day_delta = day_delta
		self.train_size = min_train_szie
		self.max_train = max_train_size
		self.test_size = test_size

		self.hf_normal = dict()
		self.nhf_normal = dict()
		self.hf_interval = dict()
		self.nhf_interval = dict()
		self.hf_fuzzy = dict()
		self.nhf_fuzzy= dict()

		self.all_label = '{0}all_label.txt'.format(SAMPLE_PATH)
		self.hf_all_normal_data = '{0}hf_all_normal.txt'.format(SAMPLE_PATH)
		self.nhf_all_normal_data = '{0}nhf_all_normal.txt'.format(SAMPLE_PATH)
		self.hf_all_interval_data = '{0}hf_all_interval.txt'.format(SAMPLE_PATH)
		self.nhf_all_interval_data = '{0}nhf_all_interval.txt'.format(SAMPLE_PATH)
		self.hf_all_fuzzy_data = '{0}hf_all_fuzzy.txt'.format(SAMPLE_PATH)
		self.nhf_all_fuzzy_data = '{0}nhf_all_fuzzy.txt'.format(SAMPLE_PATH)

		self.train_label = '{0}train_label.txt'.format(SAMPLE_PATH)
		self.hf_train_normal_data = '{0}hf_train_normal.txt'.format(SAMPLE_PATH)
		self.nhf_train_normal_data = '{0}nhf_train_normal.txt'.format(SAMPLE_PATH)
		self.hf_train_interval_data = '{0}hf_train_interval.txt'.format(SAMPLE_PATH)
		self.nhf_train_interval_data = '{0}nhf_train_interval.txt'.format(SAMPLE_PATH)
		self.hf_train_fuzzy_data = '{0}hf_train_fuzzy.txt'.format(SAMPLE_PATH)
		self.nhf_train_fuzzy_data = '{0}nhf_train_fuzzy.txt'.format(SAMPLE_PATH)

		self.test_label = '{0}test_label.txt'.format(SAMPLE_PATH)
		self.hf_test_normal_data = '{0}hf_test_normal.txt'.format(SAMPLE_PATH)
		self.nhf_test_normal_data = '{0}nhf_test_normal.txt'.format(SAMPLE_PATH)
		self.hf_test_interval_data = '{0}hf_test_interval.txt'.format(SAMPLE_PATH)
		self.nhf_test_interval_data = '{0}nhf_test_interval.txt'.format(SAMPLE_PATH)
		self.hf_test_fuzzy_data = '{0}hf_test_fuzzy.txt'.format(SAMPLE_PATH)
		self.nhf_test_fuzzy_data = '{0}nhf_test_fuzzy.txt'.format(SAMPLE_PATH)

	def organize_data(self):
		for i in range(1, 21):
			print('working:', i)
			label_file = '{0}{1}_label.txt'.format(PP_PATH, i)
			data_file = '{0}{1}_data.txt'.format(PP_PATH, i)

			hf_label, nhf_label = self.read_label(label_file)
			self.read_data(data_file, hf_label, nhf_label)

		hf_train, nhf_train, hf_test, nhf_test, hf_all, nhf_all = self.split_data()
		self.write_file(hf_train, nhf_train, hf_test, nhf_test, hf_all, nhf_all)

		print('hf size:', len(self.hf_normal))
		print('nhf size:', len(self.nhf_normal))

		print('hf train:', len(hf_train))
		print('nhf train:', len(nhf_train))

		print('hf test:', len(hf_test))
		print('nhf test:', len(nhf_test))

		print('hf all:', len(hf_all))
		print('nhf all:', len(nhf_all))


	def read_label(self, filename):
		hf_label = set()
		nhf_label = set()

		f = open(filename, 'r')
		for line in f:
			row = line.strip().split()
			label = int(row[1])
			if label == 1:
				hf_label.add(row[0])
			else:
				nhf_label.add(row[0])
		f.close()

		return hf_label, nhf_label

	def read_data(self, filename, hf_label, nhf_label):
		data_orig = dict()
		data_interval = dict()
		data_fuzzy = dict()

		f = open(filename, 'r')
		for line in f:
			row = line.strip().split()
			bid = row[0]
			date = int(row[1])
			ccs_info = set(map(int, row[2:]))

			data_orig.setdefault(bid, dict())
			data_orig[bid].setdefault(date, set())
			data_orig[bid][date] = data_orig[bid][date].union(ccs_info)

			cur_date = datetime.datetime.strptime(row[1], '%Y%m%d')
			day_delta = datetime.timedelta(days=self.day_delta)

			cur_YM = int(str(cur_date.year) + str(cur_date.month))
			delta_minus = cur_date - day_delta
			delta_plus = cur_date + day_delta
			minus_YM = int(str(delta_minus.year)+str(delta_minus.month))
			plus_YM = int(str(delta_plus.year)+str(delta_plus.month))

			data_interval.setdefault(bid, dict())
			data_interval[bid].setdefault(cur_YM, set())
			data_fuzzy.setdefault(bid, dict())
			data_fuzzy[bid].setdefault(cur_YM, set())
			data_interval[bid][cur_YM] = data_interval[bid][cur_YM].union(ccs_info)
			data_fuzzy[bid][cur_YM] = data_fuzzy[bid][cur_YM].union(ccs_info)

			if cur_YM < plus_YM and delta_plus.year < 2011:
				data_fuzzy[bid].setdefault(plus_YM, set())
				data_fuzzy[bid][plus_YM] = data_fuzzy[bid][plus_YM].union(ccs_info)
			elif cur_YM > minus_YM and delta_minus.year > 2008:
				data_fuzzy[bid].setdefault(minus_YM, set())
				data_fuzzy[bid][minus_YM] = data_fuzzy[bid][minus_YM].union(ccs_info)

		f.close()

		self._store_data(data_orig, data_interval, data_fuzzy, hf_label, nhf_label)

	def _store_data(self, data_orig, data_interval, data_fuzzy, hf_label, nhf_label):
		for bid in data_orig:
			orig_lst = data_orig[bid]
			interval_lst = data_interval[bid]
			fuzzy_lst = data_fuzzy[bid]

			temp_orig = list()
			temp_interval = list()
			temp_fuzzy = list()

			orig_sorted = sorted(list(orig_lst.keys()))
			for d in orig_sorted:
				temp_orig.append(orig_lst[d])

			interval_sorted = sorted(list(interval_lst.keys()))
			for d in interval_sorted:
				temp_interval.append(interval_lst[d])

			fuzzy_sorted = sorted(list(fuzzy_lst.keys()))
			for d in fuzzy_sorted:
				temp_fuzzy.append(fuzzy_lst[d])

			if bid in hf_label:
				self.hf_normal[bid] = temp_orig
				self.hf_interval[bid] = temp_interval
				self.hf_fuzzy[bid] = temp_fuzzy
			else:
				self.nhf_normal[bid] = temp_orig
				self.nhf_interval[bid] = temp_interval
				self.nhf_fuzzy[bid] = temp_fuzzy

	def split_data(self):
		hf_test = set(np.random.choice(len(self.hf_normal), self.test_size, replace=False))
		nhf_test = set(np.random.choice(len(self.nhf_normal), self.test_size, replace=False))

		hf_train = set(np.random.choice(len(self.hf_normal), self.train_size, replace=False))
		nhf_train = set(np.random.choice(len(self.nhf_normal), self.train_size, replace=False))

		hf_all = set(np.random.choice(len(self.hf_normal), self.max_train, replace=False))
		nhf_all = set(np.random.choice(len(self.nhf_normal), self.max_train, replace=False))

		hf_train = hf_train - hf_test - hf_all
		while len(hf_train) < self.train_size:
			print('refinding hf test')
			need_len = self.train_size - len(hf_train)
			hf_train = hf_train.union(set(np.random.choice(len(self.hf_normal), need_len, replace=False)))
			hf_train = hf_train - hf_test - hf_all

		nhf_train = (nhf_train - nhf_test) - nhf_all
		while len(nhf_train) < self.train_size:
			print('refinding nhf test')
			need_len = self.train_size - len(nhf_train)
			nhf_train = nhf_train.union(set(np.random.choice(len(self.nhf_normal), need_len, replace=False)))
			nhf_train = (nhf_train - nhf_test) - nhf_all

		return hf_train, nhf_train, hf_test, nhf_test, hf_all, nhf_all

	def write_file(self, hf_train, nhf_train, hf_test, nhf_test, hf_all, nhf_all):
		f_all_label = open(self.all_label, 'w')
		f_hf_all_normal = open(self.hf_all_normal_data, 'w')
		f_nhf_all_normal = open(self.nhf_all_normal_data, 'w')
		f_hf_all_interval = open(self.hf_all_interval_data, 'w')
		f_nhf_all_interval = open(self.nhf_all_interval_data, 'w')
		f_hf_all_fuzzy = open(self.hf_all_fuzzy_data, 'w')
		f_nhf_all_fuzzy = open(self.nhf_all_fuzzy_data, 'w')

		f_train_label = open(self.train_label, 'w')
		f_hf_train_normal = open(self.hf_train_normal_data, 'w')
		f_nhf_train_normal = open(self.nhf_train_normal_data, 'w')
		f_hf_train_interval = open(self.hf_train_interval_data, 'w')
		f_nhf_train_interval = open(self.nhf_train_interval_data, 'w')
		f_hf_train_fuzzy = open(self.hf_train_fuzzy_data, 'w')
		f_nhf_train_fuzzy = open(self.nhf_train_fuzzy_data, 'w')

		f_test_label = open(self.test_label, 'w')
		f_hf_test_normal = open(self.hf_test_normal_data, 'w')
		f_nhf_test_normal = open(self.nhf_test_normal_data, 'w')
		f_hf_test_interval = open(self.hf_test_interval_data, 'w')
		f_nhf_test_interval = open(self.nhf_test_interval_data, 'w')
		f_hf_test_fuzzy = open(self.hf_test_fuzzy_data, 'w')
		f_nhf_test_fuzzy = open(self.nhf_test_fuzzy_data, 'w')

		key_lst = list(self.hf_normal.keys())
		for i in range(len(key_lst)):
			bid = key_lst[i]

			label_line = "{0} 1\n".format(bid)
			normal_line = ''
			for ccs_lst in self.hf_normal[bid]:
				normal_line += ' '.join(map(str, ccs_lst))
				normal_line += ' -1 '
			normal_line += '-2\n'

			interval_line = ''
			for ccs_lst in self.hf_interval[bid]:
				interval_line += ' '.join(map(str, ccs_lst))
				interval_line += ' -1 '
			interval_line += '-2\n'

			fuzzy_line = ''
			for ccs_lst in self.hf_fuzzy[bid]:
				fuzzy_line += ' '.join(map(str, ccs_lst))
				fuzzy_line += ' -1 '
			fuzzy_line += '-2\n'

			if i in hf_test:
				f_test_label.write(label_line)
				f_hf_test_normal.write(normal_line)
				f_hf_test_interval.write(interval_line)
				f_hf_test_fuzzy.write(fuzzy_line)

			elif i in hf_train:
				f_train_label.write(label_line)
				f_hf_train_normal.write(normal_line)
				f_hf_train_interval.write(interval_line)
				f_hf_train_fuzzy.write(fuzzy_line)

			if i in hf_all:
				f_all_label.write(label_line)
				f_hf_all_normal.write(normal_line)
				f_hf_all_interval.write(interval_line)
				f_hf_all_fuzzy.write(fuzzy_line)

		key_lst = list(self.nhf_normal.keys())
		for i in range(len(key_lst)):
			bid = key_lst[i]

			label_line = "{0} 0\n".format(bid)
			normal_line = ''
			for ccs_lst in self.nhf_normal[bid]:
				normal_line += ' '.join(map(str, ccs_lst))
				normal_line += ' -1 '
			normal_line += '-2\n'

			interval_line = ''
			for ccs_lst in self.nhf_interval[bid]:
				interval_line += ' '.join(map(str, ccs_lst))
				interval_line += ' -1 '
			interval_line += '-2\n'

			fuzzy_line = ''
			for ccs_lst in self.nhf_fuzzy[bid]:
				fuzzy_line += ' '.join(map(str, ccs_lst))
				fuzzy_line += ' -1 '
			fuzzy_line += '-2\n'

			if i in nhf_test:
				f_test_label.write(label_line)
				f_nhf_test_normal.write(normal_line)
				f_nhf_test_interval.write(interval_line)
				f_nhf_test_fuzzy.write(fuzzy_line)
			elif i in nhf_train:
				f_train_label.write(label_line)
				f_nhf_train_normal.write(normal_line)
				f_nhf_train_interval.write(interval_line)
				f_nhf_train_fuzzy.write(fuzzy_line)

			if i in nhf_all:
				f_all_label.write(label_line)
				f_nhf_all_normal.write(normal_line)
				f_nhf_all_interval.write(interval_line)
				f_nhf_all_fuzzy.write(fuzzy_line)

		f_all_label.close()
		f_hf_all_normal.close()
		f_nhf_all_normal.close()
		f_hf_all_interval.close()
		f_nhf_all_interval.close()
		f_hf_all_fuzzy.close()
		f_nhf_all_fuzzy.close()

		f_train_label.close()
		f_hf_train_normal.close()
		f_nhf_train_normal.close()
		f_hf_train_interval.close()
		f_nhf_train_interval.close()
		f_hf_train_fuzzy.close()
		f_nhf_train_fuzzy.close()

		f_test_label.close()
		f_hf_test_normal.close()
		f_nhf_test_normal.close()
		f_hf_test_interval.close()
		f_nhf_test_interval.close()
		f_hf_test_fuzzy.close()
		f_nhf_test_fuzzy.close()


if __name__ == "__main__":
	ro = ReorganizeData(DAY_DELTA, MIN_TRAIN_SIZE, MAX_TRAIN_SIZE, TEST_SIZE)
	ro.organize_data()
















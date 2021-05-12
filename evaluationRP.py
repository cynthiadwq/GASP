import math
import numpy as np

# read in two txt file
# return two list with equal length or with less generated_patterns
def read_in_txt(ground_truth, generated_patterns):
    fileHandler = open(ground_truth, "r")
    listoflines = fileHandler.readlines()
    fileHandler.close()
    ground_truth_list = []
    patternSup = {} # a dictionary that store pattern and support count pairs
    for line in listoflines:
        spl = line.split(" ")
        # if it only contains one bucket, discard it
        if spl.count("-1") < 2:
            continue
        support = spl[-1]
        spl = spl[:-2]
        spl.append("-2")
        patternSup[(" ").join(spl)] = int(support)

    # sort patternSup by its value
    sorted_patternSup = sorted(patternSup.items(), key=lambda kv: kv[1], reverse=True)
    for key, _ in sorted_patternSup:
        ground_truth_list.append(key)

    fileHandler = open(generated_patterns, "r")
    listoflines = fileHandler.readlines()
    fileHandler.close()
    generated_patterns_list = []
    for line in listoflines:
        spl = line.split(" ")[:-1]
        # if it only contains one bucket, discard it
        if spl.count("-1") < 2:
            continue
        generated_patterns_list.append((" ").join(spl))

    return ground_truth_list, generated_patterns_list


def RecallPrecision(k, ground_truth_list, generated_patterns_list):
    ground_truth_list = ground_truth_list[:math.ceil(len(ground_truth_list) * k)]
    generated_patterns_list = generated_patterns_list[:math.ceil(len(generated_patterns_list)*k)]
    generated_patterns_set = set(generated_patterns_list)
    TP_cnt = 0
    for pattern in ground_truth_list:
        if pattern in generated_patterns_set:
            TP_cnt += 1
    print ("Among {} generated patterns, and comparing to {} ground truth patterns, {} ground truth are included in the generated pattern".format(len(generated_patterns_list), len(ground_truth_list), TP_cnt))
    return TP_cnt / len(ground_truth_list), TP_cnt / len(generated_patterns_list)

if __name__ == '__main__':
    ground_truth_list, generated_pattern_list = read_in_txt("seq_train1_CMSPADE_1.txt", "Train1GASP5M.txt")
    for k in [1]:
        recall, precision = RecallPrecision(k, ground_truth_list, generated_pattern_list)
        print ("For top {} percent of patterns, the recall is {}, the precision is {}.".format(k * 100, recall, precision))


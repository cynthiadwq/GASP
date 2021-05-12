import time
from numpy.random import choice
from random import choices
import numpy as np
import os


# read in interval_seq.txt
# output [[[1, 2],[3, 4, 5], [1, 2, 5], [4]], [[1, 2], ...]] randomly select n patients from interval_seq
def readInSeq(input):
    fileHandler = open(input)
    listOfLines = fileHandler.readlines()
    fileHandler.close()
    records = []

    # convert strings to lists
    for line in listOfLines:
        curPatient = []
        spl = line.split(" ")
        curVisit = []
        for item in spl:
            if item == "-1": # end of visit
                curPatient.append(curVisit)
                curVisit = []
            elif item == "-2": # end of patient
                curPatient.append(curVisit)
                break
            else:
                curVisit.append(int(item))
        records.append(curPatient)
    return records



def createSubseq(records, min_sup):
    start_time = time.time()
    L1 = {}
    L2 = {}
    endProb = {} # {(1, 2, 1): 0.1} prob of ending with (1, 2) sequence is 0.1
    transProb = {} # {(1, 2, 1): 0.125} prob of transition to another 1 type edge is 0.125
    lengthDist = np.zeros(3000)
    for patient in records:
        # [[1, 2],[3, 4, 5], [1, 2, 5]]
        preItem = [] # record the items in previous buckets
        numItem = sum([len(listElem) for listElem in patient]) # total number of items for this patient
        lengthDist[len(patient)] += 1
        for i in range(0, len(patient)):
            bucket = patient[i]
            # [1, 2]
            for j in range(0, len(bucket)):
                # deal with length 1 subsequence
                if bucket[j] in L1:
                    L1[bucket[j]] += 1
                else:
                    L1[bucket[j]] = 1

                # deal with length 2 subsequence, type 1(same bucket)

                numFollowBucket = numItem - len(preItem) - len(bucket)
                numSameBucket = len(bucket) - 2
                curEndProb = 1 - (numFollowBucket + numSameBucket) / numItem
                if (numSameBucket + numFollowBucket) == 0:
                    curTransProb = 0
                else:
                    curTransProb = numSameBucket / (numSameBucket + numFollowBucket)
                for k in range(j + 1, len(bucket)):
                    curTuple = (min(bucket[j], bucket[k]), max(bucket[j], bucket[k]), 1) # use min max to avoid duplicates
                    if curTuple in L2:
                        L2[curTuple] += 1
                    else:
                        L2[curTuple] = 1

                    # endProb
                    if curTuple in endProb:
                        endProb[curTuple] += curEndProb
                    else:
                        endProb[curTuple] = curEndProb
                    # transProb
                    if curTuple in transProb:
                        transProb[curTuple] += curTransProb
                    else:
                        transProb[curTuple] = curTransProb
                # deal with length 2 subsequence, type 2(different bucket)
                numFollowBucket = numItem - len(preItem) - len(bucket)
                numSameBucket = len(bucket) - 1
                curEndProb = 1 - (numFollowBucket + numSameBucket) / numItem
                if (numSameBucket + numFollowBucket) == 0:
                    curTransProb = 0
                else:
                    curTransProb = numSameBucket / (numSameBucket + numFollowBucket)

                for pre in preItem:
                    curTuple = (pre, bucket[j], 2)
                    if curTuple in L2:
                        L2[curTuple] += 1
                    else:
                        L2[curTuple] = 1
                    # endProb
                    if curTuple in endProb:
                        endProb[curTuple] += curEndProb
                    else:
                        endProb[curTuple] = curEndProb
                    # transProb
                    if curTuple in transProb:
                        transProb[curTuple] += curTransProb
                    else:
                        transProb[curTuple] = curTransProb
            preItem = preItem + bucket

    # prune L2
    pruned_L2 = {}
    safe_L1 = set()
    for key, value in L2.items():
        if value >= min_sup:
            pruned_L2[key] = value
            safe_L1.add(key[0])
            safe_L1.add(key[1])

    totalDegree = 0
    for key in safe_L1:
        totalDegree += L1[key]

    # deal with start probability
    startNode = []
    startProb = []
    for key, value in L1.items():
        if key in safe_L1:
            startNode.append(key)
            startProb.append(value / totalDegree)

    # normalize endProb
    for key, value in endProb.items():
        endProb[key] = value / L2[key]

    # normalize tranProb
    for key, value in transProb.items():
        transProb[key] = value / L2[key]

    # normalize lengthDist
    lengthDist = lengthDist / len(records)
    lengthProb = {}
    count = 0
    agg = 0
    for num in lengthDist:
        lengthProb[count] = agg + num
        agg += num
        count += 1
        if agg >= 1:
            break

    print("--- %s seconds ---" % (time.time() - start_time))
    return pruned_L2, startNode, startProb, endProb, transProb, lengthProb

def createGraph(L2, startNode, startProb, endProb, transProb, lengthProb):
    start_time = time.time()
    edgeW = {}
    # add edges
    for key, value in L2.items():
        # add edge
        if key[2] == 2:
            if key[0] in edgeW:
                edgeW[key[0]].append((key[1], 2, value))
            else:
                edgeW[key[0]] = [(key[1], 2, value)]
        if key[2] == 1:
            if key[0] in edgeW:
                edgeW[key[0]].append((key[1], 1, value))
            else:
                edgeW[key[0]] = [(key[1], 1, value)]
            if key[1] in edgeW:
                edgeW[key[1]].append((key[0], 1, value))
            else:
                edgeW[key[1]] = [(key[0], 1, value)]

    pathFreq = {}
    pathWeight = {}
    # A random walk with 10000000 iterations
    for i in range(10000000):

        # choose a starting vertex
        vertex = choice(startNode, 1, p = startProb)[0] # return a list of length 1, choose the first item
        path = []
        bucket = [str(vertex)]
        curTransProb = [0.5, 0.5] # initialize current transition probability
        numItem = 1 # number of items in the current sequence
        seqWeight = 0 # total weight of the current sequence
        numBucket = 0
        while True:
            # no more child node, end sequence
            if vertex not in edgeW:
                break

            vertex_successor = [n for n in edgeW[vertex]]

            vertex_successor_tmp = []
            for item in vertex_successor:
                if (item[1] == 1) and (str(item[0]) in bucket):
                    # the successor with type 1 link to current item already exists in current bucket
                    continue
                else:
                    vertex_successor_tmp.append(item)
            if len(vertex_successor_tmp) == 0:
                break
            # randomly choose a child node based on weight
            prob = np.zeros(len(vertex_successor_tmp))
            vertex_successor = []
            totalWeight = 0
            cnt = 0
            for tuple in vertex_successor_tmp:
                type = tuple[1]
                value = tuple[2]
                curWeight = value * curTransProb[type - 1]
                vertex_successor.append(tuple)
                prob[cnt] = curWeight
                cnt += 1
                totalWeight += curWeight
            if totalWeight == 0:
                break
            prob = prob / totalWeight # normalize prob
            vertex_next = choices(vertex_successor, prob)[0]

            # process bucket
            if vertex_next[1] == 1:  # same bucket
                bucket.append(str(vertex_next[0]))
            else:  # transfer to next bucket
                bucket.sort()
                path = path + bucket
                path.append("-1")
                bucket = [str(vertex_next[0])]
                numBucket += 1
            numItem += 1
            seqWeight += vertex_next[2]

            # decide if current sequence ends
            endingProb = 0
            if numItem in lengthProb:
                endingProb = lengthProb[numItem]
                if vertex_next[1] == 2:  # type 2 link
                    endingProb += endProb[(vertex, vertex_next[0], 2)]
                else:
                    endingProb += endProb[(min(vertex, vertex_next[0]), max(vertex, vertex_next[0]), 1)]
                endingProb = endingProb / 2
            else:
                endingProb = 1
            curEndProb = [endingProb, 1 - endingProb]
            if choice([1, 0], 1, p=curEndProb) == 1:
                # end sequence
                break

            # update curTransProb
            if vertex_next[1] == 2: # type 2 link, the order between parent and child cannot be changed
                tmp = transProb[(vertex, vertex_next[0], 2)]
                curTransProb = [tmp, 1 - tmp]
            else: # type 1 link, can change order
                tmp = transProb[(min(vertex, vertex_next[0]), max(vertex, vertex_next[0]), 1)]
                curTransProb = [tmp, 1 - tmp]

            vertex = vertex_next[0]

        if numBucket < 1:
            continue

        bucket.sort()
        path = path + bucket
        path.append("-1")
        path.append("-2")
        strpath = " ".join(x for x in path)

        if strpath in pathFreq:
            pathFreq[strpath] += 1
            pathWeight[strpath] += seqWeight
        else:
            pathFreq[strpath] = 1
            pathWeight[strpath] = seqWeight

    sorted_x = sorted(pathWeight.items(), key=lambda x: x[1], reverse=True)
    fileHandler = open("Train1GASP10M.txt", "w+")
    for key, value in sorted_x:
        if pathFreq[key] > 1:
            fileHandler.write(str(key))
            fileHandler.write(" " + str(value))
            fileHandler.write("\n")
    fileHandler.close()

    print("--- %s seconds ---" % (time.time() - start_time))





if __name__ == '__main__':
    records = readInSeq("seq_train1.txt")
    L2, startNode, startProb, endProb, transProb, lengthProb = createSubseq(records, 71)
    createGraph(L2, startNode, startProb, endProb, transProb, lengthProb)

import numpy as np
from collections import defaultdict
import itertools

""" acuracy metrics """


def MAE(predictions):
    __, __, true_r, predict_r = list(zip(*predictions))
    return np.mean(abs(np.array(true_r) - np.array(predict_r)))


def RMSE(predictions):
    __, __, true_r, predict_r = list(zip(*predictions))
    return np.sqrt(np.mean(np.sqrt(np.array(true_r) - np.array(predict_r))))


def hit_rate(topN, predictions):
    total = 0
    hits = 0
    for pred in predictions:
        lo_uid = pred[0]
        lo_iid = pred[1]
        hit = False
        for iid, predict_rating in topN[int(lo_uid)]:
            if int(lo_iid) == int(iid):
                # print('--item hit:', lo_iid, 'user:',lo_uid)
                hit = True
                break
        if hit:
            hits += 1
        total += 1
    # print('hits:', hits, ',', lo_uid, lo_iid)
    return hits/total

    def cummulative_hit_rate(topN, predictions, rating_cut_off=0):
        total = 0
        hits = 0
        for pred in predictions:
            lo_uid = pred[0]
            lo_iid = pred[1]
            actual_rating = pred[2]
            if actual_rating > rating_cut_off:
                hit = False
                for iid, predict_rating in topN[int(lo_uid)]:
                    if int(lo_iid) == int(iid):
                        # print('--item hit:', lo_iid, 'user:',lo_uid)
                        hit = True
                        break
                if hit:
                    hits += 1
                total = +1
        # print('hits:', hits, ',' lo_uid, lo_iid)
        return hits/total

    def ARHR(topN, predictions):
        total = 0
        hits = 0
        for pred in predictions:
            hit_rank = 0
            rank = 0
            lo_uid = pred[0]
            lo_iid = pred[1]
            for iid, predict_rating in topN[int(lo_uid)]:
                rank += 1
                if int(lo_uid) == int(iid):
                    # print('--item hit:', lo_uid, 'user:', lo_uid,'rank',rank)
                    hit_rank = rank
                    break
                if hit_rank > 0:
                    hits += 1.0 / int(hit_rank)
                total += 1
        return hits/total

    def ratings_hit_rate(topN, predictions):
        total = defaultdict(float)
        hits = defaultdict(float)
        for pred in predictions:
            lo_uid = pred[0]
            lo_iid = pred[1]
            actual_rating = pred[2]
            hit = False
            for iid, predict_rating in topN[int(lo_uid)]:
                if int(lo_uid) == int(iid):
                    hit = True
                    break
            if hit:
                hits[actual_rating] += 1
            total[actual_rating] += 1
        # print ('hits:', hits, ',', lo_uid,lo_iid)
        for rating in sorted(hits.keys()):
            print(rating, hits[rating]/total[rating])


''' Beyond metrics to be used with full anti test set evaluation '''


def spread(topN, predictions):
    P = defaultdict(float)
    total = sum([len(i) for i in topN.values()])
    for pred_uid, pred_iid, _, _ in predictions:
        for iid, predict_rating in topN[int(pred_uid)]:
            if int(iid) == int(pred_iid):
                P[int(iid)] += 1.0/total
    return -1.0 * sum([p * np.log(p) for p in P.values()])


def coverage(predictions, test_set):
    return len(predictions) / len(test_set)


def diversity(topN, sims_algo):
    n = 0
    total = 0
    sim_mat = sims_algo.compute_similarities()
    for user_id in topN.keys():
        pairs = itertools.combinations(topN[user_id], 2)
        for pair in pairs:
            item_1 = pair[0][0]
            item_2 = pair[1][0]
            iid_1 = sims_algo.train_set.to_inner_iid(str(item_1))
            iid_2 = sims_algo.train_set.to_inner_iid(str(item_2))
            total += sim_mat[iid_1][iid_2]
            n += 1
    return 1 - (total/n)


def novelty(topN, rankings):
    n = 0
    total = 0
    for uid in topN.keys():
        for iid, _ in topN[uid]:
            rank = rankings[iid]
            total += rank
            n += 1
    return total / n

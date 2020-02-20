"""
3 classes - User collaborative filtering, Item-collaborative iltering
and simple collaborativefiltering.

simple inherit from process.py, metrics.py

user-CF, item-CF have no inheritance
"""

from process import Movielens
from process import Evaluation
import metrics

from surprise import KNNBasic
from collections import defaultdict
import heapq
import time

# get similarity matrix


def get_Sim_matrix(train_set, name, user_based=False):
    # Build a similarity matrix using surprise lib's KNNbasic Module.
    sim_options = {'name': name, 'user_based': user_based}
    model = KNNBasic(sim_options=sim_options)
    model.fit(train_set)
    return model.sim


def scorefunc(rating, SimScore):
    return rating * SimScore


# USER BASED CF
class User_CF():
    def __init__(self, train_set, train_set_looxv, test_set_looxv, similarity):
        self.train_set, self.train_set_looxv, = train_set, train_set_looxv
        self.test_set_looxv = test_set_looxv
        self.similarity = similarity

        t0 = time.time()
        self.sims_matrix_looxv = get_Sim_matrix(
            train_set_looxv, similarity, True)
        t1 = time.time()
        self.topN = self.predict()
        t2 = time.time()
        print('User matrix', t1-t0, 'User TopN:', t2-t1)

    def predict(self, n=8):
        # Build Top List Based on X-Validated Training Data
        topN = defaultdict(list)
        k = 20
        for uiid in range(self.train_set_looxv.n_users):
            # Get k most similar users
            similarity_row = self.sims_matrix_looxv[uiid]
            similarity_users = []
            for inne_Id, score, in enumerate(similarity_row):
                if inne_Id != uiid:
                    similarity_users.append((inne_Id, score))
            KNeighbors = heapq.nlargest(
                k, similarity_users, key=lambda x: x[1])

            # Build candidiate List
            candidiates = defaultdict(float)
            candidiates_weight = defaultdict(float)
            for inner_Id, user_sim_score in KNeighbors:
                other_ratings = self.train_set_looxv.ur[inne_Id]
                for ratings in other_ratings:
                    candidiates[ratings[0]
                                ] += scorefunc(ratings[1], user_sim_score)
                    candidiates[ratings[0]] += user_sim_score

            # Build Seen List
            has_watched = [item[0] for item in self.train_set_looxv.ur[uiid]]

            # produce TopN from candidiate dict less has_watched
            pos = 0
            for item_Id, score in sorted(candidiates.items(),
                                         key=lambda x: x[1], reverse=True):

                if not has_watched.__contains__(int(item_Id)):

                    movie_Id = self.train_set_looxv.to_raw_iid(item_Id)
                    topN[int(self.train_set_looxv.to_raw_uid(uiid))
                         ].append((int(movie_Id), score))
                    # print (m1.getMovieName(int(movie_Id))),
                    # score/candidiates_weight[item_Id])
                    pos += 1
                    if pos > n:

                        break
        return topN

# ITEM BASED CF


class ItemCF():
    def __init__(self, train_set, train_set_looxv, test_set_looxv, similarity, size):
        self.train_set, self.train_set_looxv = train_set, train_set_looxv
        self.test_set_looxv = test_set_looxv
        self.similarity = similarity
        self.model_size = size

        t0 = time.time()
        self.sims_matrix_looxv = get_Sim_matrix(
            train_set_looxv, similarity, False)
        t1 = time.time()
        self.topN = self.predict()
        t2 = time.time()
        print('Item Matrix:', t1-t0, 'Item TopN:', t2-t1)

    def predict(self, n=8):
        # Build Top list based on X-validated Training data
        topN = defaultdict(list)
        k = 20
        for uiid in range(self.train_set_looxv.n_users):
            # Get k top rated items
            active_user_ratings = self.train_set_looxv.ur[uiid]
            KNeighbors = heapq.nlargest(k, active_user_ratings, lambda x: x[1])

            # Build candidiate List
            candidiates = defaultdict(float)
            candidiates_weight = defaultdict(float)

            for iiid, rating in KNeighbors:
                # for each top item rated, get all similar items
                similarity_row = self.sims_matrix_looxv[iiid]
                if self.model_size:
                    # implement 'model size Truncation
                    similarity_row = heapq.nlargest(
                        self.model_size, similarity_row)
                for other_iiid, item_sim_score in enumerate(similarity_row):
                    candidiates[other_iiid] += scorefunc(
                        rating, item_sim_score)
                    candidiates_weight[other_iiid] += item_sim_score

                # Build seen List
                has_watched = [item[0]
                               for item in self.train_set_looxv.ur[uiid]]

                # produce TopN from Candiaite dict less haswatched
                pos = 0
                for item_id, score in sorted(candidiates.items(),
                                             key=lambda x: x[1], reverse=True):
                    if not has_watched.__contains__(int(item_id)):
                        movie_id = self.train_set_looxv.to_raw_iid(item_id)
                        topN[int(self.train_set_looxv.to_raw_uid(uiid))
                             ].append((int(movie_id), score))
                        # print (m1.getmovieName(int(movie_id)),
                        # score/candidiates_weight[item_id])
                        pos += 1
                        if pos > n:
                            break
        return topN


# Simple CF
class simple_CF():
    def __init__(self, MovielensObject):
        self.ml = MovielensObject
        self.data = self.ml.load1Mdata()
        self.train_set, self.train_set_looxv, self.test_set_looxv = self.processData(
            self.data)
        self.test_user_inner_id = self.test_user_summary('56')

    def run_user_CF(self, similarity):
        user_cf = User_CF(self.train_set, self.train_set_looxv,
                          self.test_set_looxv, similarity)
        return user_cf.predict()

    def run_item_cf(self, similarity, model_size=None):
        item_cf = ItemCF(self.train_set, self.train_set_looxv,
                         self.test_set_looxv, similarity, model_size)
        return item_cf.predict()

    def process_data(self, data):
        print('preparing data....')
        eval = Evaluation(data)
        return eval.train_set, eval.LOOX_trainSet, eval.LOOX_testSet

    def test_user_summary(self, test_user):
        test_user_inner_id = self.train_set.to_inner_id(test_user)
        print("Target User Total ratings": len(self.train_set.ur[test_user_inner_id]))
        print('Target user 5 star Ratings:')
        for iid, rating in self.train_set.ur[test_user_inner_id]:
            if rating == 5.0:
                print(ml.get_movie_name(int(self.train_set.to_raw_iid(iid))))
        return test_user_inner_id

    def recommendation_metrics(self, topN, test_user):
        print('-hit_rate:', metrics.hit_rate(topN, self.test_set_looxv))
        print('- rating_hit_rate:',
              metrics.rating_hit_rate(topN, self.test_set_looxv))
        # print('cummulative hit rate:', metrics.cummulative_hit_rate(topN, self.test_set_looxv))
        # print(' - AverageReciprocal hitRate:', metrics.ARHR(topN, self.test_set_looxv))
        print('Target user Top 8 List:')
        counter = 0
        for movie_id, score in topN[int(test_user)]:
            print(ml.get_movie_name(int(movie_id)))
            counter += 1
            if counter > 7:
                break


t = time.time()
ml = Movielens()
rankings = ml.get_popularity_rating()
cf = simple_CF(ml)
print('Load data:', time.time() - t)

user = cf.run_user_CF('cosine')


pearson = cf.run_item_cf('pearson')
cosine = cf.run_item_cf('cosine')
msd = cf.run_item_cf('msd')


model_size_10 = cf.run_item_cf('cosine', model_size=10)
model_size_30 = cf.run_item_cf('cosine', model_size=30)
model_size_50 = cf.run_item_cf('cosine', model_size=50)

cf.recommendation_metrics(model_size_10, '56')
cf.recommendation_metrics(user, '56')

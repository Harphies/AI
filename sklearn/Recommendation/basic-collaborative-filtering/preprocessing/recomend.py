"""
3 classes - User collaborative filtering, Item-collaborative iltering
and simple collaborativefiltering.

simple inherit from process.py, metrics.py

user-CF, item-CF have no inheritance
"""

from process import Movielens
from process import Evaluation

from surprise import KNNBasic
from collections import defaultdict
import heapq
import time

# get similarity matrix


def get_Sim_Matrix(train_set, name, user_based=False):
    # Build a similarity matrix using surprise lib's KNNbasic Module.
    sim_options = {'name': name, 'user_based': user_based}
    model = KNNBasic(sim_options=sim_options)
    model.fit(train_set)
    return model.sim


def scorefunction(rating, SimScore):
    return rating * SimScore


# USER BASED CF

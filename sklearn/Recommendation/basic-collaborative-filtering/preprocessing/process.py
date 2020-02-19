"""
scripts to preprocess the movielens data

Create a class named Movielens, with functions load_data, 
get_genre, get_year,
get_movie_name, get_movie_id
get_popularity, get_users_ratings.

"""

from surprise import Dataset, Reader
import re
import csv
from collections import defaultdict
from surprise.model_selection import train_test_split, LeaveOneOut


class Movielens:
    movieID_to_name = {}
    name_to_movieID = {}
    ratingsPath = 'ml-latest-small/ratings.csv'
    moviePath = 'ml-latest-small/movies.csv'

    def load1Mdata(self):
        reader = Reader(line_format='user item rating timestamp',
                        sep='::', skip_lines=1)
        ratings_Dataset = Dataset.load_from_file(
            'ml-1m/ratings.dat', reader=reader)
        with open('ml-1m/movies.dat', newline='', encoding='ISO-8859-1') as df:
            for row in df:
                row = row.split('::')
                movie_Id, movie_Name = int(row[0]), row[1]
                self.movieID_to_name[movie_Id] = movie_Name
                self.name_to_movieID[movie_Name] = movie_Id
                del movie_Id, movie_Name
        return ratings_Dataset

    def loadData(self):
        reader = Reader(line_format='user item rating timestamp',
                        sep=',', skip_lines=1)
        ratings_Dataset = Dataset.load_from_file(
            self.ratingsPath, reader=reader)
        with open(self.moviePath, newline='', encoding='ISO-8859-1') as cf:
            movie_Reader = csv.reader(cf)
            next(movie_Reader)  # below row without this
            for row in movie_Reader:
                movie_Id, movie_Name = int(row[0]), row[1]
                self.movieID_to_name[movie_Id] = movie_Name
                self.name_to_movieID[movie_Name] = movie_Id
                del movie_Id, movie_Name
        return ratings_Dataset

    def get_users_ratings(self, user):
        # All ratings for a given user, list of tuples
        user_ratings = []
        hit_user = False
        with open(self.ratingsPath, newline='') as cv:
            rating_reader = csv.reader(cv)
            next(rating_reader)  # below errors without this
            for row in rating_reader:
                # print(row)
                user_Id = int(row[0])
                if user == user_Id:
                    movide_Id = int(row[1])
                    rating = float(row[2])
                    user_ratings.append(movide_Id, rating)
                    hit_user = True
                    if(hit_user and (user != user_Id)):
                        break
        return user_ratings

    def get_popularity_rating(self):
        # pupularity rank = most rated movies
        ratings = defaultdict(int)
        rankings = defaultdict(int)
        with open(self.ratingsPath, newline='') as cf:
            ratings_reader = csv.reader(cf)
            next(ratings_reader)
            for row in ratings_reader:
                movie_Id = int(row[1])
                ratings[movie_Id] += 1
        rank = 1
        for movie_Id, rating_count in sorted(ratings.items(),
                                             key=lambda x: x[1], reverse=True):
            rankings[movie_Id] = rank
            rank += 1
        return rankings

    def get_genre(self):
        # genre dictionary and integer coded
        genres = defaultdict(list)
        genre_Ids = {}
        max_genre_Id = 0
        with open(self.moviePath, newline='', encoding='ISO-8859-1') as cv:
            movie_reader = csv.reader(cv)
            next(movie_reader)
            for row in movie_reader:
                movie_Id = int(row[0])
                genre_list = row[2].split('|')
                genre_Id_list = []
                for genre in genre_list:
                    if genre in genre_Ids:
                        genre_Id = genre_Ids[genre]
                    else:
                        genre_Id = max_genre_Id
                        genre_Ids[genre] = genre_Id
                        max_genre_Id += 1
                    genre_Id_list.append(genre_Id)
                genres[movie_Id] = genre_Id_list

            for (movie_Id, genre_Id_list) in genres.items():
                one_hot = [0]*max_genre_Id
                for genre_Id in genre_Id_list:
                    one_hot[genre_Id] = 1
                genres[movie_Id] = one_hot
            return genres

    def get_year(self):
        # Year of the movie
        years = defaultdict(int)
        p = re.compile(r"(?:\((\d{4})\))?\s*$")
        with open(self.moviePath, newline='', encoding='ISO-8859-1') as cf:
            movie_reader = csv.reader(cf)
            next(movie_reader)
            for row in movie_reader:
                movie_Id = int(row[0])
                title = row[1]
                year = p.search(title).group(1)
                if year:
                    years[movie_Id] = int(year)
        return years

    def get_movie_name(self, movie_Id):
        return self.movieID_to_name[movie_Id]

    def get_movie_Id(self, movie_Name):
        return self.name_to_movieID[movie_Name]


class Evaluation:
    def __init__(self, data):
        self.train_set, self.test_set = train_test_split(
            data, test_size=0.25, random_state=1)

        LOOX = LeaveOneOut(1, random_state=1)
        for x_train, x_test in LOOX.split(data):
            self.LOOX_trainSet = x_train
            self.LOOX_testSet = x_test
            del x_test, x_train
        self.LOOX_anti_testSet = self.LOOX_trainSet.build_anti_testset()

        self.full_trainSet = data.buid_full_trainset()
        self.full_anti_testSet = self.full_trainSet.build_anti_testset()

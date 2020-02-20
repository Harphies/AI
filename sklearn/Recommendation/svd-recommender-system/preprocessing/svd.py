''' Evryone want to be a beast until it's time to do what beasts do'''

from surprise import SVD, NormalPredictor
from surprise.model_selection import GridSearchCV
from util import MovieLens
from evaluator import Evaluator


def load_movie_lens_data():
    ml = MovieLens()
    data = ml.loadData()
    rankings = ml.getPopularityRanking()
    return ml, data, rankings


ml, data, rankings = load_movie_lens_data()

# Build evaluation object
evaluator = Evaluator(data, rankings)

# Grid Search Tuning
param_grid = {'n_factors': [50, 100, 150], 'n_epoch': [
    25, 50, 75], 'lr_all': [0.005, 0.01], 'reg_all': [0.02, 0.1, 0.5]}
gs = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=3)
gs.fit(data)


# Build tuned SVD, untuned SVD, random models
params = gs.best_params['rmse']
svd_tuned = SVD(reg_all=params['reg_all'], n_factors=params['n_factor'],
                n_epochs=params['n_epochs'], lr_all=params['lr_all'])

svd = SVD()
random = NormalPredictor()

# Add models to evaluation objects
evaluator.addModel(svd_tuned, 'SVDtuned')
evaluator.addModel(svd, 'SVD')
evaluator.addModel(random, 'random')

# Evaluate objects = fit models , build topN, lists, run prediction/hit rate based metrics.

evaluator.evaluateModel(True)

# Build topN list for target user 56
evaluator.sampleUser(ml)

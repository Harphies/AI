# import the libraries
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# features needed
column_names = ['user_id', 'item_id', 'rating', 'timestamp']
df = pd.read_csv('u.data', sep='\t', names=column_names)

df.head()

# let's get the movie title
movie_titles = pd.read_csv("Movie_Id_Titles")
movie_titles.head()

# let's merge the data
df = pd.merge(df, movie_titles, on='item_id')
df.head()

# let's explore the data a little to get more understanding of the data
sns.set_style('white')

# let's create a rating dataframe with average rating and number of rating
df.groupby('title')['rating'].mean().sort_values(ascending=False).head()

df.groupby('title')['rating'].count().sort_values(ascending=False).head()

# create a dataframe for the means
ratings = pd.DataFrame(df.groupby('title')['rating'].mean())
ratings.head()

ratings['num of ratings'] = pd.DataFrame(df.groupby('title')['rating'].count())
ratings.head()

# let's plot some histogram

# let's plot for num of ratings
plt.figure(figsize=(10, 4))
ratings['num of ratings'].hist(bins=70)

# let's plot for ratings
plt.figure(figsize=(10, 4))
ratings['rating'].hist(bins=70)


# let's have a joint plot for both num of rating and ratingss
sns.jointplot(x='rating', y='num of ratings', data=ratings, alpha=0.5)

# recommender Agorithm
moviemat = df.pivot_table(index='user_id', columns='title', values='rating')
moviemat.head()

# sort all the movies
ratings.sort_values('num of ratings', ascending=False).head(10)

# let's choose a movie to recomend based on
ratings.head()

# let's grab user ratings for some movies
starwars_user_ratings = moviemat['Star Wars (1977']
liarliar_user_ratings = moviemat['Liar Liar (1997)']
starwars_user_ratings.head()

# we can use pandas corrwith function to check the correlation
similar_to_startwar = moviemat.corrwith(starwars_user_ratings)
similar_to_liarliar = moviemat.corrwith(liarliar_user_ratings)

# let's remove the NaN data
corr_starwars = pd.DataFrame(similar_to_startwar, columns=['Correlation'])
corr_starwars.dropna(inplace=True)
corr_starwars.head()

# sort the values base on  correlation
corr_starwars.sort_values('Correlation', ascending=False).head(10)

# let's make the correlation a little advance
corr_starwars = corr_starwars.join(ratings['num of ratings'])
corr_starwars.head()

# let's increase the sorting critera
corr_starwars[corr_starwars['num of ratings'] > 100].sort_values(
    'Correlation', ascending=False).head()


# let's do for another movie
corr_liarliar = pd.DataFrame(similar_to_liarliar, columns=['Correlation'])
corr_liarliar.dropna(inplace=True)
corr_liarliar = corr_liarliar.join(ratings['num of ratings'])
corr_liarliar[corr_liarliar['num of ratings'] > 100].sort_values(
    'Correlation', ascending=False).head()

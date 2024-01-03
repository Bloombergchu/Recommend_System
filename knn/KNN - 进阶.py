import os
import time
import gc
import argparse

# data science imports
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
# utils import
from fuzzywuzzy import fuzz

####
#Splitthe datasetinto train and test data
from sklearn.model_selection import train_test_split
#Building and training the model
from sklearn.neighbors import KNeighborsClassifier
#k-Fold Cross-Validation
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV


class KnnRecommender:
    """
    This is an item-based collaborative filtering recommender with
    KNN implmented by sklearn
    """
    def __init__(self, path_movies, path_ratings):
        """
        Recommender requires path to data: movies data and ratings data
        Parameters
        ----------
        path_movies: str, movies data file path
        path_ratings: str, ratings data file path
        """
        self.path_movies = path_movies
        self.path_ratings = path_ratings
        self.movie_rating_thres = 0
        self.user_rating_thres = 0
        self.model = NearestNeighbors()

        self.train_data, self.test_data = self._prep_data()




    def set_filter_params(self, movie_rating_thres, user_rating_thres):
        """
        set rating frequency threshold to filter less-known movies and
        less active users
        Parameters
        ----------
        movie_rating_thres: int, minimum number of ratings received by users
        user_rating_thres: int, minimum number of ratings a user gives
        """
        self.movie_rating_thres = movie_rating_thres
        self.user_rating_thres = user_rating_thres

    def set_model_params(self, n_neighbors, algorithm, metric, n_jobs=None):
        """
        set model params for sklearn.neighbors.NearestNeighbors
        Parameters
        ----------
        n_neighbors: int, optional (default = 5)
        algorithm: {'auto', 'ball_tree', 'kd_tree', 'brute'}, optional
        metric: string or callable, default 'minkowski', or one of
            ['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan']
        n_jobs: int or None, optional (default=None)
        """
        if n_jobs and (n_jobs > 1 or n_jobs == -1):
            os.environ['JOBLIB_TEMP_FOLDER'] = '/tmp'
        self.model.set_params(**{
            'n_neighbors': n_neighbors,
            'algorithm': algorithm,
            'metric': metric,
            'n_jobs': n_jobs})

    def _prep_data(self):
        """
        prepare data for recommender
        1. movie-user scipy sparse matrix
        2. hashmap of movie to row index in movie-user scipy sparse matrix
        """
        # read data
        df_movies = pd.read_csv(
            os.path.join(self.path_movies),
            usecols=['movieId', 'title'],
            dtype={'movieId': 'int32', 'title': 'str'})
        df_ratings = pd.read_csv(
            os.path.join(self.path_ratings),
            usecols=['userId', 'movieId', 'rating'],
            dtype={'userId': 'int32', 'movieId': 'int32', 'rating': 'float32'})
        # filter data
        df_movies_cnt = pd.DataFrame(
            df_ratings.groupby('movieId').size(),
            columns=['count'])
        popular_movies = list(set(df_movies_cnt.query('count >= @self.movie_rating_thres').index))  # noqa
        movies_filter = df_ratings.movieId.isin(popular_movies).values

        df_users_cnt = pd.DataFrame(
            df_ratings.groupby('userId').size(),
            columns=['count'])
        active_users = list(set(df_users_cnt.query('count >= @self.user_rating_thres').index))  # noqa
        users_filter = df_ratings.userId.isin(active_users).values

        df_ratings_filtered = df_ratings[movies_filter & users_filter]

        # pivot and create movie-user matrix
        movie_user_mat = df_ratings_filtered.pivot(
            index='movieId', columns='userId', values='rating').fillna(0)
        # create mapper from movie title to index
        '''
        hashmap = {
            movie: i for i, movie in
            enumerate(list(df_movies.set_index('movieId').loc[movie_user_mat.index].title)) # noqa
        }
        # transform matrix to scipy sparse matrix
        '''
        movie_user_mat_sparse = csr_matrix(movie_user_mat.values)

        # clean up
        del df_movies, df_movies_cnt, df_users_cnt
        del df_ratings, df_ratings_filtered, movie_user_mat
        gc.collect()
#########這裡隨機分成 train70% 和 test30%  random隨機選一下
        train_data,test_data = movie_user_mat_sparse.randomSplit([0.7, 0.3])
        return train_data, test_data  #是個矩陣

    def Cross_validation(self): #參考42行-52行
        #在train70%裡面執行
        knn2 = KNeighborsClassifier()
        param_grid = {'n_neighbors': np.arange(1, 25)}
        knn_gscv = GridSearchCV(knn2, param_grid, cv=5)

        # fit model to data
        # 這裡的X,Y 得從movie_user_mat_sparse 上面的處理數據得到的裡面 提出來

        knn_gscv.fit(self.train_data)
        # check top performing n_neighbors value 不怎么需要
        knn_gscv.best_params_
        # check mean score for the top performing value of n_neighbors
        knn_gscv.best_score_
        #這個字典的value值
        k = knn_gscv.best_score_.value
        return k
 ###########結果返回給self


    def _fuzzy_matching(self, hashmap, fav_movie):
        """
        return the closest match via fuzzy ratio.
        If no match found, return None
        Parameters
        ----------
        hashmap: dict, map movie title name to index of the movie in data
        fav_movie: str, name of user input movie
        Return
        ------
        index of the closest match
        """
        match_tuple = []
        # get match
        for title, idx in hashmap.items():
            ratio = fuzz.ratio(title.lower(), fav_movie.lower())
            if ratio >= 60:
                match_tuple.append((title, idx, ratio))
        # sort
        match_tuple = sorted(match_tuple, key=lambda x: x[2])[::-1]
        if not match_tuple:
            print('Oops! No match is found')
        else:
            print('Found possible matches in our database: '
                  '{0}\n'.format([x[0] for x in match_tuple]))
            return match_tuple[0][1]





    def _inference(self, model, data, hashmap,
                   fav_movie, n_recommendations):
        """
        return top n similar movie recommendations based on user's input movie
        Parameters
        ----------
        model: sklearn model, knn model
        data: movie-user matrix
        hashmap: dict, map movie title name to index of the movie in data
        fav_movie: str, name of user input movie
        n_recommendations: int, top n recommendations
        Return
        ------
        list of top n similar movie recommendations
        """
        # fit
        model.fit(data)
        # get input movie index
        print('You have input movie:', fav_movie)
        idx = self._fuzzy_matching(hashmap, fav_movie)
        # inference
        print('Recommendation system start to make inference')
        print('......\n')
        t0 = time.time()
        distances, indices = model.kneighbors(
            data[idx],
            n_neighbors=n_recommendations+1)
        # get list of raw idx of recommendations
        raw_recommends = \
            sorted(
                list(
                    zip(
                        indices.squeeze().tolist(),
                        distances.squeeze().tolist()
                    )
                ),
                key=lambda x: x[1]
            )[:0:-1]
        print('It took my system {:.2f}s to make inference \n\
              '.format(time.time() - t0))
        # return recommendation (movieId, distance)
        return raw_recommends

    def make_recommendations(self, fav_movie, n_recommendations):
        """
        make top n movie recommendations
        Parameters
        ----------
        fav_movie: str, name of user input movie
        n_recommendations: int, top n recommendations
        """
        # get data
        movie_user_mat_sparse, hashmap = self._prep_data()
        # get recommendations
        raw_recommends = self._inference(
            self.model, movie_user_mat_sparse, hashmap,
            fav_movie, n_recommendations)
        # print results
        reverse_hashmap = {v: k for k, v in hashmap.items()}
        print('Recommendations for {}:'.format(fav_movie))
        for i, (idx, dist) in enumerate(raw_recommends):
            print('{0}: {1}, with distance '
                  'of {2}'.format(i+1, reverse_hashmap[idx], dist))


def parse_args():
    parser = argparse.ArgumentParser(
        prog="Movie Recommender",
        description="Run KNN Movie Recommender")
    parser.add_argument('--path', nargs='?', default='../KNN/Data',
                        help='input data path')
    parser.add_argument('--movies_filename', nargs='?', default='movies.csv',
                        help='provide movies filename')
    parser.add_argument('--ratings_filename', nargs='?', default='ratings.csv',
                        help='provide ratings filename')
    parser.add_argument('--movie_name', nargs='?', default='',
                        help='provide your favoriate movie name')
    parser.add_argument('--top_n', type=int, default=10,
                        help='top n movie recommendations')
    return parser.parse_args()





if __name__ == '__main__':
    # get args
    args = parse_args()
    data_path = args.path
    movies_filename = args.movies_filename
    ratings_filename = args.ratings_filename
    movie_name = args.movie_name
    top_n = args.top_n
    # initial Recommence system

    recommender = KnnRecommender(
        os.path.join(data_path, movies_filename),
        os.path.join(data_path, ratings_filename))
    # set params 42-52 ，把（50,50）賦值到裡面
    recommender.set_filter_params(50, 50)

    #這個20就是KNN裡面的K,也就是需要交叉驗證需要找到的最合適的K
    #所以要變成變量，然後帶到這個裡面
    #再实例化,为了使用最合适的K值
    Myrecommender = KnnRecommender
    best_K = Myrecommender.Cross_validation
    recommender.set_model_params(best_K, 'brute', 'cosine', -1)
    # make recommendations 建模
    recommender.make_recommendations(movie_name, top_n)

######测试数据集已经准备好
    train_set, test_set = Myrecommender._prep_data
    #开始测试数据
    ##show first 5 model predictions on the test data
    Myrecommender.predict(test_set)[0:5]
    # check accuracy of our model on the test data
    Myrecommender.score(train_set, test_set)



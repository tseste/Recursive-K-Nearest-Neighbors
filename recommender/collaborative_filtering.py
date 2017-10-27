"""The recommender class model."""
import csv
import pandas as pd

from .utilities import timer


class Preprocessing:
    """Data preprocessing for KNNRegressor and RecursiveKNNRegressor."""

    @staticmethod
    def preprocess_train(train_csv, group_by, other):
        """Preprocessing of the train data.

        :param train_csv: The name of the train data csv to read (str)
        :param group_by: The column which the data will be indexed (str)
        :param other: The column which the data will be grouped by (str)
        :return: The grouped DataFrame of the train data (DataFrameGroupBy)
        """
        train = pd.read_csv(train_csv)

        train.set_index(group_by, inplace=True)

        train = train.groupby(other)

        return train

    @staticmethod
    def preprocess_test(test_csv):
        """Preprocessing of the test data.

        :param test_csv: The name of the test data csv to read (str)
        :return: The DataFrame of the test data (DataFrame)
        """
        return pd.read_csv(test_csv)

    @staticmethod
    def preprocess_similarity(sim_csv):
        """Preprocessing od the similarity data.

        :param sim_csv: The name of the similarities data csv to read (str)
        :return: The grouped DataFrame of the similarities (DataFrameGroupBy)
        """
        similarity = pd.read_csv(sim_csv)

        similarity = similarity.loc[similarity.similarity > 0]

        sim_col_1 = similarity.columns[0]
        sim_col_2 = similarity.columns[1]
        sim_col_3 = similarity.columns[2]

        similarity = similarity.append(
                        similarity.rename(
                            columns={
                                sim_col_1: sim_col_2,
                                sim_col_2: sim_col_1
                            }
                        )
                    )

        similarity.set_index(sim_col_2, inplace=True)

        similarity.sort_values(by=sim_col_3, kind='mergesort', ascending=False,
                               inplace=True)

        similarity = similarity.groupby(sim_col_1)

        return similarity


class KNNRegressor(Preprocessing):
    """The k-nearest neighbors regression class."""

    def __init__(self, neighbors, group_by='user'):
        """Init KNN."""
        assert type(neighbors) is list, 'neighbors parameter must be a list'
        assert type(group_by) is str, 'group_by must be string'

        if group_by not in ['user', 'item']:
            raise ValueError('group_by must be one of "user" or "item"')

        other = 'item'
        if group_by == other:
            other = 'user'

        self.train = None
        self.test = None
        self.similarity = None
        self.__predictions_csv = None
        self.__no_nn_csv = None
        self.__log_csv = None

        self.neighbors = neighbors
        self.neighbors.sort()
        self.__max_neighbors = self.neighbors[-1]

        self.group_by = group_by
        self.__other = other

        self.__no_nn_counter = 0
        self.__u_m_not_in_train_counter = 0
        self.__similarity_not_found_counter = 0

    def fit(self, train_csv, sim_csv):
        """Fit the train data to the model."""
        start = timer()
        assert type(train_csv) is str, 'train_csv must be string'
        assert type(sim_csv) is str, 'sim_csv must be string'
        self.train = self.preprocess_train(train_csv,
                                           self.group_by,
                                           self.__other)
        self.similarity = self.preprocess_similarity(sim_csv)

        sim_csv = sim_csv.split('/')[-1]
        self.__predictions_csv = 'predictions_{}'.format(sim_csv)
        self.__no_nn_csv = 'no_nn_{}'.format(sim_csv)
        self.__log_csv = 'log_{}'.format(sim_csv)
        timer(start)

    def _pair_neighbors(self, user_item, item_user):
        try:
            item_user_ratings = self.train.get_group(item_user)
        except:
            self.__u_m_not_in_train_counter += 1
            return None

        try:
            user_item_similarities = self.similarity.get_group(user_item)
        except:
            self.__similarity_not_found_counter += 1
            return None

        related_neighbors = user_item_similarities.loc[
                              user_item_similarities.index.isin(
                                item_user_ratings.index)]

        if related_neighbors.empty:
            self.__no_nn_counter += 1
        return related_neighbors

    def _pair_predictions(self, related_neighbors, related_ratings):
        predictions = []
        numerator = 0
        denominator = 0
        iteration = 0
        for u_m, similarity in zip(related_neighbors.index.values,
                                   related_neighbors.similarity.values):

            numerator += similarity * related_ratings.rating.at[u_m]
            denominator += similarity

            iteration += 1
            if iteration in self.neighbors:
                predictions.append(numerator / denominator)

            if iteration == self.__max_neighbors:
                break

        if len(predictions) < len(self.neighbors):
            extend = len(self.neighbors) - len(predictions)
            predictions += extend * [numerator/denominator]

        return predictions

    def predict(self, test_csv):
        """Predict the test values."""
        if self.train is None or self.similarity is None:
            raise AttributeError('You must run fit before predict')

        assert type(test_csv) is str, 'test_csv must be string'
        self.test = self.preprocess_test(test_csv)

        start = timer()

        with open(self.__predictions_csv, 'w') as pf:
            pred_w = csv.writer(pf)
            pred_w.writerow([self.group_by, self.__other] + self.neighbors)

            with open(self.__no_nn_csv, 'w') as no_nn_f:
                no_nn_w = csv.writer(no_nn_f)
                no_nn_w.writerow([self.group_by, self.__other, 'rating'])

                loop_test = zip(self.test[self.group_by].values,
                                self.test[self.__other].values,
                                self.test.rating.values)

                for user_item, item_user, rating in loop_test:

                    related_neighbors = self._pair_neighbors(user_item,
                                                             item_user)
                    if related_neighbors is not None:
                        if not related_neighbors.empty:
                            related_ratings = self.train.get_group(item_user)
                            predictions = self._pair_predictions(
                                                            related_neighbors,
                                                            related_ratings)
                            pred_w.writerow([user_item, item_user] +
                                            predictions)
                        else:
                            no_nn_w.writerow([user_item, item_user, rating])

        with open(self.__log_csv, 'w') as mf:
            log_w = csv.writer(mf)
            log_w.writerow(['no_nn', 'not_in_train',
                            'no_similarity_found', 'time'])
            log_w.writerow([self.__no_nn_counter,
                            self.__u_m_not_in_train_counter,
                            self.__similarity_not_found_counter,
                            timer(start, True)])
        timer(start)


class RecursiveKNNRegressor(Preprocessing):
    """The Recursive k-nearest neighbors regression class."""

    def __init__(self, neighbors, recursive_neighbors, group_by='user'):
        """Init RecursiveKNN."""
        assert type(neighbors) is list, 'neighbors parameter must be a list'
        assert type(recursive_neighbors) is list, \
            'recursive_neighbors parameter must be a list'
        assert type(group_by) is str, 'group_by must be string'

        if group_by not in ['user', 'item']:
            raise ValueError('group_by must be one of "user" or "item"')

        other = 'item'
        if group_by == other:
            other = 'user'

        self.train = None
        self.test = None
        self.similarity = None
        self.__predictions_csv = None
        self.__no_nn_csv = None
        self.__log_csv = None

        self.neighbors = neighbors
        self.neighbors.sort()
        self.__max_neighbors = self.neighbors[-1]

        self.group_by = group_by
        self.__other = other

        self.__no_nn_counter = 0

        self.recursive_neighbors = recursive_neighbors
        self.recursive_neighbors.sort()
        self.__max_recursive_neighbors = self.recursive_neighbors[-1]

    def fit(self, train_csv, sim_csv):
        """Fit the train data to the model."""
        start = timer()
        assert type(train_csv) is str, 'train_csv must be string'
        assert type(sim_csv) is str, 'sim_csv must be string'
        self.train = self.preprocess_train(train_csv,
                                           self.group_by,
                                           self.__other)
        self.similarity = self.preprocess_similarity(sim_csv)

        sim_csv = sim_csv.split('/')[-1]
        self.__predictions_csv = 'recursive_predictions_{}'.format(sim_csv)
        self.__no_nn_csv = 'recursive_no_nn_{}'.format(sim_csv)
        self.__log_csv = 'recursive_log_{}'.format(sim_csv)
        timer(start)

    def _search_predict_recursive_neighbors(self, user_item, item_user):
        neighbors_predictions = []
        related_ratings = self.train.get_group(item_user)

        related_neighbors = self.similarity.get_group(user_item)

        found = 0
        for neighbor in related_neighbors.index.values:
            recursive_neighbors = self.similarity.get_group(neighbor)

            recursive_neighbors = recursive_neighbors.loc[
                recursive_neighbors.index.isin(related_ratings.index)]

            if not recursive_neighbors.empty:
                found += 1
                predictions = self._recursive_pair_predictions(
                                                    recursive_neighbors,
                                                    related_ratings)
                # one recursive user for with m recursive_neighbors predictions
                neighbors_predictions.append([neighbor, item_user] +
                                             predictions)

            if found == self.__max_neighbors:
                break
        # preprocess found recursive
        neighbors_predictions = pd.DataFrame(neighbors_predictions,
                                             columns=[self.group_by,
                                                      self.__other] +
                                             self.recursive_neighbors)
        neighbors_predictions.set_index(self.group_by, inplace=True)
        return neighbors_predictions

    def _recursive_pair_predictions(self, related_neighbors, related_ratings):
        predictions = []
        numerator = 0
        denominator = 0
        iteration = 0
        for u_m, similarity in zip(related_neighbors.index.values,
                                   related_neighbors.similarity.values):

            numerator += similarity * related_ratings.rating.at[u_m]
            denominator += similarity

            iteration += 1
            if iteration in self.recursive_neighbors:
                predictions.append(numerator / denominator)

            if iteration == self.__max_recursive_neighbors:
                break

        if len(predictions) < len(self.recursive_neighbors):
            extend = len(self.recursive_neighbors) - len(predictions)
            predictions += extend * [numerator / denominator]

        return predictions

    def _pair_predictions(self, related_neighbors, related_ratings):
        predictions = []
        numerator = 0
        denominator = 0
        iteration = 0
        for u_m, similarity in zip(related_neighbors.index.values,
                                   related_neighbors.similarity.values):

            numerator += similarity * related_ratings.at[u_m]
            denominator += similarity

            iteration += 1
            if iteration in self.neighbors:
                predictions.append(numerator / denominator)

            if iteration == self.__max_neighbors:
                break

        if len(predictions) < len(self.neighbors):
            extend = len(self.neighbors) - len(predictions)
            predictions += extend * [numerator / denominator]

        return predictions

    def predict(self, no_nn_csv):
        """Predict the test values."""
        if self.train is None or self.similarity is None:
            raise AttributeError('You must run fit before predict')

        assert type(no_nn_csv) is str, 'no_nn_csv must be string'
        self.test = self.preprocess_test(no_nn_csv)

        start = timer()

        with open(self.__predictions_csv, 'w') as pf:
            pred_w = csv.writer(pf)
            pred_w.writerow(['recursive_neighbors', self.group_by,
                             self.__other] + self.neighbors)

            with open(self.__no_nn_csv, 'w') as no_nn_f:
                no_nn_w = csv.writer(no_nn_f)
                no_nn_w.writerow([self.group_by, self.__other, 'rating'])

                loop_test = zip(self.test[self.group_by].values,
                                self.test[self.__other].values,
                                self.test.rating.values)

                for user_item, item_user, rating in loop_test:
                    related_ratings = self._search_predict_recursive_neighbors(
                                                        user_item, item_user)
                    if not related_ratings.empty:
                        # the function .isin() from DataFrame is not needed
                        # because I already know they are in from
                        # the _search_predict_recursive_neighbors so
                        # I just loc the similarities
                        related_neighbors = self.similarity.get_group(
                                        user_item).loc[related_ratings.index]
                        for m in self.recursive_neighbors:
                            predictions = self._pair_predictions(
                                related_neighbors, related_ratings.loc[:, m])
                            pred_w.writerow(
                                [m, user_item, item_user] + predictions)
                    else:
                        self.__no_nn_counter += 1
                        no_nn_w.writerow([user_item, item_user, rating])

        with open(self.__log_csv, 'w') as mf:
            log_w = csv.writer(mf)
            log_w.writerow(['no_nn', 'time'])
            log_w.writerow([self.__no_nn_counter, timer(start, True)])
        timer(start)

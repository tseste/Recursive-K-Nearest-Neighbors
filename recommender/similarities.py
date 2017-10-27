"""Compute similarities of users or items in a dataset."""
import csv
import numpy as np

from collections import deque

from .utilities import timer, csv_to_dictionary


class Similarities:
    """Similarity formulas wrapper."""

    @staticmethod
    def cosine_similarity(train_csv, group_by):
        """Compute and write the cosine similarity in a file.

        Args:
            train_csv(str): The train file name to read
            group_by(str): The grouping parameter 'user' or 'item'
        """
        train = csv_to_dictionary(train_csv, group_by)

        start = timer()

        unique_u_m = deque(sorted(train))

        u_m_norm = dict.fromkeys(train)
        for u_m in u_m_norm:
            u_m_norm[u_m] = np.linalg.norm(list(train[u_m].values()))

        sim_file = group_by + '_cosine_similarity.csv'
        with open(sim_file, 'w') as sf:
            w = csv.writer(sf)
            w.writerow([group_by + '_1', group_by + '_2', 'similarity'])

            for u_m_1 in unique_u_m.copy():
                unique_u_m.popleft()
                for u_m_2 in unique_u_m:
                    common_u_m = set(train[u_m_1]).intersection(train[u_m_2])
                    if common_u_m:
                        xy = 0
                        for common in common_u_m:
                            xy += train[u_m_1][common] * train[u_m_2][common]
                        sim = xy/(u_m_norm[u_m_1] * u_m_norm[u_m_2])
                        w.writerow([u_m_1, u_m_2, sim])

        timer(start)

    @staticmethod
    def cosine_similarity_modified(train_csv, group_by):
        """Compute and write the modified cosine similarity in a file.

        Args:
            train_csv(str): The train file name to read
            group_by(str): The grouping parameter 'user' or 'item'
        """
        train = csv_to_dictionary(train_csv, group_by)

        start = timer()

        unique_u_m = deque(sorted(train))

        sim_file = group_by + '_modified_cosine_similarity.csv'
        with open(sim_file, 'w') as sf:
            w = csv.writer(sf)
            w.writerow([group_by + '_1', group_by + '_2', 'similarity'])

            for u_m_1 in unique_u_m.copy():
                unique_u_m.popleft()
                for u_m_2 in unique_u_m:
                    common_u_m = set(train[u_m_1]).intersection(train[u_m_2])
                    if common_u_m:
                        xy = 0
                        norm_1 = np.linalg.norm([train[u_m_1][common]
                                                for common in common_u_m])
                        norm_2 = np.linalg.norm([train[u_m_2][common]
                                                for common in common_u_m])
                        for common in common_u_m:
                            xy += train[u_m_1][common] * train[u_m_2][common]
                        sim = xy/(norm_1 * norm_2)
                        w.writerow([u_m_1, u_m_2, sim])

        timer(start)

    @staticmethod
    def adjusted_cosine_similarity(train_csv, group_by):
        """Compute and write the adjusted cosine similarity in a file.

        Args:
            train_csv(str): The train file name to read
            group_by(str): The grouping parameter 'user' or 'item'
        """
        other = 'item'
        if group_by == other:
            other = 'user'

        users = csv_to_dictionary(train_csv, group_by)
        items = csv_to_dictionary(train_csv, other)

        start = timer()

        u_m_avg = dict.fromkeys(items)
        for item in items:
            u_m_avg[item] = (sum(items[item].values()) /
                             len(items[item].values()))

        unique_u_m = deque(sorted(users))

        sim_file = group_by + '_adjusted_cosine_similarity.csv'
        with open(sim_file, 'w') as sf:
            w = csv.writer(sf)
            w.writerow([group_by + '_1', group_by + '_2', 'similarity'])

            for u_m_1 in unique_u_m.copy():
                unique_u_m.popleft()
                for u_m_2 in unique_u_m:
                    common_items = set(users[u_m_1]).intersection(users[u_m_2])
                    if common_items:
                        sum_xy = 0
                        sum_sqx = 0
                        sum_sqy = 0
                        for common in common_items:
                            rx_meanx = users[u_m_1][common] - u_m_avg[common]
                            ry_meany = users[u_m_2][common] - u_m_avg[common]
                            sum_xy += rx_meanx * ry_meany
                            sum_sqx += rx_meanx**2
                            sum_sqy += ry_meany**2
                        if sum_xy:
                            sim = sum_xy/np.sqrt(sum_sqx*sum_sqy)
                            w.writerow([u_m_1, u_m_2, sim])

        timer(start)

    @staticmethod
    def adjusted_cosine_similarity_modified(train_csv, group_by):
        """Compute and write the modified adjusted cosine similarity in a file.

        Args:
            train_csv(str): The train file name to read
            group_by(str): The grouping parameter 'user' or 'item'
        """
        other = 'item'
        if group_by == other:
            other = 'user'

        users = csv_to_dictionary(train_csv, group_by)
        items = csv_to_dictionary(train_csv, other)

        start = timer()

        unique_u_m = deque(sorted(users))

        u_m_avg = dict.fromkeys(items)
        for item in items:
            u_m_avg[item] = (sum(items[item].values()) /
                             len(items[item].values()))

        user_norm = dict.fromkeys(users)
        for user in users:
            user_norm[user] = np.sqrt(sum(
                                    [(users[user][item] - u_m_avg[item])**2
                                     for item in users[user]]))

        sim_file = group_by + '_modified_adjusted_cosine_similarity.csv'
        with open(sim_file, 'w') as sf:
            w = csv.writer(sf)
            w.writerow([group_by + '_1', group_by + '_2', 'similarity'])

            for u_m_1 in unique_u_m.copy():
                unique_u_m.popleft()
                for u_m_2 in unique_u_m:
                    common_items = set(users[u_m_1]).intersection(users[u_m_2])
                    if common_items:
                        sum_xy = 0
                        for common in common_items:
                            rx_meanx = users[u_m_1][common] - u_m_avg[common]
                            ry_meany = users[u_m_2][common] - u_m_avg[common]
                            sum_xy += rx_meanx * ry_meany
                        if sum_xy:
                            sim = sum_xy/(user_norm[u_m_1] * user_norm[u_m_2])
                            w.writerow([u_m_1, u_m_2, sim])

        timer(start)

    @staticmethod
    def pearson_correlation_coefficient(train_csv, group_by):
        """Compute and write the pearson correlation coefficient in a file.

        Args:
            train_csv(str): The train file name to read
            group_by(str): The grouping parameter 'user' or 'item'
        """
        train = csv_to_dictionary(train_csv, group_by)

        start = timer()

        unique_u_m = deque(sorted(train))

        u_m_avg = dict.fromkeys(train)
        for u_m in u_m_avg:
            u_m_avg[u_m] = sum(train[u_m].values()) / len(train[u_m].values())

        sim_file = group_by + '_pearson_correlation_coefficient.csv'
        with open(sim_file, 'w') as sf:
            w = csv.writer(sf)
            w.writerow([group_by + '_1', group_by + '_2', 'similarity'])

            for u_m_1 in unique_u_m.copy():
                unique_u_m.popleft()
                for u_m_2 in unique_u_m:
                    common_u_m = set(train[u_m_1]).intersection(train[u_m_2])
                    if common_u_m:
                        sum_xy = 0
                        sum_sq_x_xbar = 0
                        sum_sq_y_ybar = 0
                        for common in common_u_m:
                            rx_meanx = train[u_m_1][common] - u_m_avg[u_m_1]
                            ry_meany = train[u_m_2][common] - u_m_avg[u_m_2]
                            sum_xy += rx_meanx * ry_meany
                            sum_sq_x_xbar += rx_meanx**2
                            sum_sq_y_ybar += ry_meany**2
                        if sum_xy:
                            sim = sum_xy/np.sqrt(sum_sq_x_xbar * sum_sq_y_ybar)
                            w.writerow([u_m_1, u_m_2, sim])

        timer(start)

    @staticmethod
    def pearson_correlation_coefficient_modification_1(train_csv, group_by):
        """Compute and write the modified pearson correlation coefficient in a file.

        Args:
            train_csv(str): The train file name to read
            group_by(str): The grouping parameter 'user' or 'item'
        """
        train = csv_to_dictionary(train_csv, group_by)

        start = timer()

        unique_u_m = deque(sorted(train))

        sim_file = group_by + '_modified_pearson_correlation_coefficient_1.csv'
        with open(sim_file, 'w') as sf:
            w = csv.writer(sf)
            w.writerow([group_by + '_1', group_by + '_2', 'similarity'])

            for u_m_1 in unique_u_m.copy():
                unique_u_m.popleft()
                for u_m_2 in unique_u_m:
                    common_u_m = set(train[u_m_1]).intersection(train[u_m_2])
                    if common_u_m:
                        sum_xy = 0
                        sum_sq_x_xbar = 0
                        sum_sq_y_ybar = 0
                        common_1 = 0
                        common_2 = 0
                        for common in common_u_m:
                            common_1 += train[u_m_1][common]
                            common_2 += train[u_m_2][common]
                        avg_1 = common_1/len(common_u_m)
                        avg_2 = common_2/len(common_u_m)
                        for common in common_u_m:
                            rx_meanx = train[u_m_1][common] - avg_1
                            ry_meany = train[u_m_2][common] - avg_2
                            sum_xy += rx_meanx * ry_meany
                            sum_sq_x_xbar += rx_meanx**2
                            sum_sq_y_ybar += ry_meany**2
                        if sum_xy:
                            sim = sum_xy/np.sqrt(sum_sq_x_xbar * sum_sq_y_ybar)
                            w.writerow([u_m_1, u_m_2, sim])

        timer(start)

    @staticmethod
    def pearson_correlation_coefficient_modification_2(train_csv, group_by):
        """Compute and write the modified pearson correlation coefficient in a file.

        Args:
            train_csv(str): The train file name to read
            group_by(str): The grouping parameter 'user' or 'item'
        """
        train = csv_to_dictionary(train_csv, group_by)

        start = timer()

        unique_u_m = deque(sorted(train))

        u_m_avg = dict.fromkeys(train)
        for u_m in u_m_avg:
            u_m_avg[u_m] = sum(train[u_m].values()) / len(train[u_m].values())

        u_m_norm = dict.fromkeys(train)
        for u_m in u_m_norm:
            u_m_norm[u_m] = np.sqrt(sum([(train[u_m][m_u] - u_m_avg[u_m])**2
                                        for m_u in train[u_m]]))

        sim_file = group_by + '_modified_pearson_correlation_coefficient_2.csv'
        with open(sim_file, 'w') as sf:
            w = csv.writer(sf)
            w.writerow([group_by + '_1', group_by + '_2', 'similarity'])

            for u_m_1 in unique_u_m.copy():
                unique_u_m.popleft()
                for u_m_2 in unique_u_m:
                    common_u_m = set(train[u_m_1]).intersection(train[u_m_2])
                    if common_u_m:
                        sum_xy = 0
                        for common in common_u_m:
                            rx_meanx = train[u_m_1][common] - u_m_avg[u_m_1]
                            ry_meany = train[u_m_2][common] - u_m_avg[u_m_2]
                            sum_xy += rx_meanx * ry_meany
                        if sum_xy:
                            sim = sum_xy/(u_m_norm[u_m_1] * u_m_norm[u_m_2])
                            w.writerow([u_m_1, u_m_2, sim])

        timer(start)

    @staticmethod
    def mean_squared_difference(train_csv, group_by):
        """Compute and write the mean squared difference in a file.

        Args:
            train_csv(str): The train file name to read
            group_by(str): The grouping parameter 'user' or 'item'
        """
        train = csv_to_dictionary(train_csv, group_by)

        start = timer()

        unique_u_m = deque(sorted(train))

        sim_file = group_by + '_mean_squared_difference.csv'
        with open(sim_file, 'w') as sf:
            w = csv.writer(sf)
            w.writerow([group_by + '_1', group_by + '_2', 'similarity'])

            for u_m_1 in unique_u_m.copy():
                unique_u_m.popleft()
                for u_m_2 in unique_u_m:
                    common_u_m = set(train[u_m_1]).intersection(train[u_m_2])
                    if common_u_m:
                        xy = 0
                        for common in common_u_m:
                            xy += (train[u_m_1][common] -
                                   train[u_m_2][common])**2
                        if xy:
                            sim = len(common_u_m)/xy
                            w.writerow([u_m_1, u_m_2, sim])

        timer(start)

    @staticmethod
    def mean_absolute_difference(train_csv, group_by):
        """Compute and write the mean absolute difference in a file.

        Args:
            train_csv(str): The train file name to read
            group_by(str): The grouping parameter 'user' or 'item'
        """
        train = csv_to_dictionary(train_csv, group_by)

        start = timer()

        unique_u_m = deque(sorted(train))

        sim_file = group_by + '_mean_absolute_difference.csv'
        with open(sim_file, 'w') as sf:
            w = csv.writer(sf)
            w.writerow([group_by + '_1', group_by + '_2', 'similarity'])

            for u_m_1 in unique_u_m.copy():
                unique_u_m.popleft()
                for u_m_2 in unique_u_m:
                    common_u_m = set(train[u_m_1]).intersection(train[u_m_2])
                    if common_u_m:
                        xy = 0
                        for common in common_u_m:
                            xy += abs(train[u_m_1][common] -
                                      train[u_m_2][common])
                        if xy:
                            sim = len(common_u_m)/xy
                            w.writerow([u_m_1, u_m_2, sim])

        timer(start)

    @staticmethod
    def jaccard_coefficient(train_csv, group_by):
        """Compute and write the  jaccard coefficient in a file.

        Args:
            train_csv(str): The train file name to read
            group_by(str): The grouping parameter 'user' or 'item'
        """
        train = csv_to_dictionary(train_csv, group_by)

        start = timer()

        unique_u_m = deque(sorted(train))

        sim_file = group_by + '_jaccard_coefficient.csv'
        with open(sim_file, 'w') as sf:
            w = csv.writer(sf)
            w.writerow([group_by + '_1', group_by + '_2', 'similarity'])

            for u_m_1 in unique_u_m.copy():
                unique_u_m.popleft()
                for u_m_2 in unique_u_m:
                    common_u_m = len(set(train[u_m_1]).intersection(
                                           train[u_m_2]))
                    if common_u_m:
                        u_m_union = len(set(train[u_m_1]).union(train[u_m_2]))
                        sim = common_u_m/u_m_union
                        w.writerow([u_m_1, u_m_2, sim])

        timer(start)

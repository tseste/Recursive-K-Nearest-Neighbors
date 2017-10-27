"""Evaluators for the recommender model."""
import numpy as np
import pandas as pd


class Evaluator:
    """RMSE, RMSUE, MAE, MAUE evaluators wrapper."""

    def _preprocessing(self, test_file, predictions_file):
        test = pd.read_csv(test_file)
        predictions = pd.read_csv(predictions_file)
        test = test.set_index(['user', 'item']).sort_index()
        predictions = predictions.set_index(['user', 'item']).sort_index()
        test = test.loc[test.index.isin(predictions.index)]
        test_values = test.values
        return test_values, predictions

    @staticmethod
    def _predictions_counter(n_pred, n_r_pred, pred_file):
        pred_counter = {
            'knn': [n_pred],
            'r_knn': [n_r_pred],
            'total': [n_pred + n_r_pred]
        }
        pd.DataFrame(pred_counter).to_csv('counter_' + pred_file, index=False)

    @staticmethod
    def _rmse_func(y_true, y_pred, recursive=False):
        if recursive:
            y_pred = y_pred.drop('recursive_neighbors', axis=1)
        subtract = y_true - y_pred
        sq_sum_div = (subtract**2).sum(axis=0) / y_pred.shape[0]
        return np.sqrt(sq_sum_div)

    @staticmethod
    def _total_rmse(rmse_file, recursive_rmse_file, n1, n2):
        """Combine the rmse values of KNN and RecursiveKNN."""
        rmse = pd.read_csv(rmse_file)
        recursive_rmse = pd.read_csv(recursive_rmse_file)
        # loc the same columns of knn ['1', '3', ...]
        recursive_rmse_same = recursive_rmse.loc[:, rmse.columns]
        # (rmse^2)*number_of_predictions
        rmse_sq_n1 = (rmse**2)*n1
        # (recursive_rmse^2)*number_of_recursive_predictions
        recursive_rmse_sq_n2 = (recursive_rmse_same**2)*n2
        # duplicate rmse shape as recursive shape for vector computations
        rmse_sq_n1_shape_recursive = pd.concat(
                                    [rmse_sq_n1]*recursive_rmse_sq_n2.shape[0],
                                    ignore_index=True)
        total = np.sqrt(
                (rmse_sq_n1_shape_recursive + recursive_rmse_sq_n2)/(n1 + n2))
        f_total = total.join(recursive_rmse['recursive_neighbors'])
        f_total.to_csv('total_' + rmse_file, index=False)

    def rmse(self, test_file=None, pred_file=None,
             r_test_file=None, r_pred_file=None):
        """Compute and write the rmse and recursive rmse of the predictions."""
        if test_file and pred_file:
            test_values, predictions = self._preprocessing(test_file,
                                                           pred_file)
            pred_file = pred_file.split('/')[-1]
            n_pred = predictions.shape[0]
            rmse = self._rmse_func(test_values, predictions)

            rmse.to_frame().T.to_csv('rmse_{}'.format(pred_file),
                                     index=False)

        if r_test_file and r_pred_file:
            test_values, predictions = self._preprocessing(r_test_file,
                                                           r_pred_file)
            r_pred_file = r_pred_file.split('/')[-1]
            predictions = predictions.groupby('recursive_neighbors')
            get_first_group = list(predictions.groups.keys())[0]
            n_r_pred = predictions.get_group(get_first_group).shape[0]

            recursive_rmse = predictions.apply(
                lambda group_predictions: self._rmse_func(test_values,
                                                          group_predictions,
                                                          True))

            recursive_rmse.to_csv('rmse_{}'.format(r_pred_file))
        if test_file and pred_file and r_test_file and r_pred_file:
            self._predictions_counter(n_pred, n_r_pred, pred_file)
            rmse_file = 'rmse_' + pred_file
            r_rmse_file = 'rmse_' + r_pred_file
            self._total_rmse(rmse_file, r_rmse_file, n_pred, n_r_pred)

    @staticmethod
    def _rmsue_func(y_true, y_pred, group_by, recursive=False):
        if recursive:
            y_pred = y_pred.drop('recursive_neighbors', axis=1)
        subtract = y_true - y_pred
        group_predictions = subtract.reset_index().groupby(group_by)
        # calculate each user or item rmse
        user_rmse = group_predictions.apply(lambda user: np.sqrt(
            (user.drop(['user', 'item'], axis=1)**2).sum(axis=0) /
            user.shape[0]))
        # average rmses and return
        return user_rmse.sum(axis=0)/user_rmse.shape[0]

    def rmsue(self, group_by, test_file=None, pred_file=None,
              r_test_file=None, r_pred_file=None):
        """Compute and write rmsue and recursive rmsue of the predictions."""
        if test_file and pred_file:
            test_values, predictions = self._preprocessing(test_file,
                                                           pred_file)
            pred_file = pred_file.split('/')[-1]
            rmsue = self._rmsue_func(test_values, predictions, group_by)
            rmsue.to_frame().T.to_csv('rmsue_{}'.format(pred_file),
                                      index=False)
        if r_test_file and r_pred_file:
            test_values, predictions = self._preprocessing(r_test_file,
                                                           r_pred_file)
            r_pred_file = r_pred_file.split('/')[-1]
            predictions = predictions.groupby('recursive_neighbors')
            recursive_rmsue = predictions.apply(
                lambda group_predictions: self._rmsue_func(test_values,
                                                           group_predictions,
                                                           group_by,
                                                           True))
            recursive_rmsue.to_csv('rmsue_{}'.format(r_pred_file))

    @staticmethod
    def _mae_func(y_true, y_pred, recursive=False):
        if recursive:
            y_pred = y_pred.drop('recursive_neighbors', axis=1)
        subtract = y_true - y_pred
        return abs(subtract).sum(axis=0) / y_pred.shape[0]

    @staticmethod
    def _total_mae(mae_file, recursive_mae_file, n1, n2):
        """Combine the rmse values of KNN and RecursiveKNN."""
        mae = pd.read_csv(mae_file)
        recursive_mae = pd.read_csv(recursive_mae_file)
        # loc the same columns of knn ['1', '3', ...]
        recursive_mae_same = recursive_mae.loc[:, mae.columns]
        # mae*number_of_predictions
        mae_n1 = mae*n1
        # (recursive_mae)*number_of_recursive_predictions
        recursive_mae_n2 = recursive_mae_same*n2
        # duplicate rmse shape as recursive shape for vector computations
        mae_n1_shape_recursive = pd.concat(
                                    [mae_n1]*recursive_mae_n2.shape[0],
                                    ignore_index=True)
        total = (mae_n1_shape_recursive + recursive_mae_n2)/(n1 + n2)
        f_total = total.join(recursive_mae['recursive_neighbors'])
        f_total.to_csv('total_' + mae_file, index=False)

    def mae(self, test_file=None, pred_file=None,
            r_test_file=None, r_pred_file=None):
        """Compute and write the mae and recursive mae of the predictions."""
        if test_file and pred_file:
            test_values, predictions = self._preprocessing(test_file,
                                                           pred_file)
            pred_file = pred_file.split('/')[-1]
            n_pred = predictions.shape[0]
            mae = self._mae_func(test_values, predictions)
            mae.to_frame().T.to_csv('mae_{}'.format(pred_file), index=False)

        if r_test_file and r_pred_file:
            test_values, predictions = self._preprocessing(r_test_file,
                                                           r_pred_file)
            r_pred_file = r_pred_file.split('/')[-1]
            predictions = predictions.groupby('recursive_neighbors')
            get_first_group = list(predictions.groups.keys())[0]
            n_r_pred = predictions.get_group(get_first_group).shape[0]

            recursive_mae = predictions.apply(
                lambda group_predictions: self._mae_func(test_values,
                                                         group_predictions,
                                                         True))

            recursive_mae.to_csv('mae_{}'.format(r_pred_file))
            if test_file and pred_file and r_test_file and r_pred_file:
                self._predictions_counter(n_pred, n_r_pred, pred_file)
                mae_file = 'mae_' + pred_file
                r_mae_file = 'mae_' + r_pred_file
                self._total_mae(mae_file, r_mae_file, n_pred, n_r_pred)

    @staticmethod
    def _maue_func(y_true, y_pred, group_by, recursive=False):
        if recursive:
            y_pred = y_pred.drop('recursive_neighbors', axis=1)
        subtract = y_true - y_pred
        group_predictions = subtract.reset_index().groupby(group_by)
        # calculate each user or item rmse
        user_mae = group_predictions.apply(lambda user: abs(
            user.drop(['user', 'item'], axis=1)).sum(axis=0) / user.shape[0])
        # average maes and return
        return user_mae.sum(axis=0)/user_mae.shape[0]

    def maue(self, group_by, test_file=None, pred_file=None,
             r_test_file=None, r_pred_file=None):
        """Compute and write the maue and recursive maue of the predictions."""
        if test_file and pred_file:
            test_values, predictions = self._preprocessing(test_file,
                                                           pred_file)
            pred_file = pred_file.split('/')[-1]
            maue = self._maue_func(test_values, predictions, group_by)
            maue.to_frame().T.to_csv('maue_{}'.format(pred_file), index=False)

        if r_test_file and r_pred_file:
            test_values, predictions = self._preprocessing(r_test_file,
                                                           r_pred_file)
            r_pred_file = r_pred_file.split('/')[-1]
            predictions = predictions.groupby('recursive_neighbors')
            recursive_maue = predictions.apply(
                lambda group_predictions: self._maue_func(test_values,
                                                          group_predictions,
                                                          group_by,
                                                          True))
            recursive_maue.to_csv('maue_{}'.format(r_pred_file))

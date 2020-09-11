import random
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import LinearRegression, Lasso


def get_real_labels():
    pathlist = Path("D:\\dtu_experiment_data\\all_test_samples").glob('*')
    labels_as_list = []
    for grade_folder in list(pathlist):
        diagram_files = Path(grade_folder).glob('*')
        for diagram_file in diagram_files:
            match = re.search("songid-(\\d+(?:\\.\\d+)?)_start-date-(\d+-\d+-\d+).png", diagram_file.name)
            if (match is not None and len(match.groups()) == 2):
                labels_as_list.append([int(float(match.group(1))), str(match.group(2)), int(grade_folder.name)])
            else:
                raise Exception("Incorrect filename")
    label_df = pd.DataFrame(
        data=labels_as_list,
        columns=["id", "train_start_date", "grade"]
    )
    return label_df


def get_experiment_prediction_and_scores(name):
    root_folder = "D:\\dtu_experiment_data"
    data_df = pd.read_csv(root_folder +
                          "\\" + name +
                          "\\all_test_data_ids_and_scores.csv",
                          sep=' *, *')
    return data_df


class OutlieScoreFeatureExtractor:

    def get_data(self, name):
        root_folder = "D:\\dtu_experiment_data"
        experiment_folders = Path(root_folder).glob('*')
        data_df = None
        for experiment_folder in experiment_folders:
            if (experiment_folder.name == "all_test_samples"):
                continue
            temp_data_df = pd.read_csv(experiment_folder / "all_test_data_ids_and_scores.csv", sep=' *, *')
            if (data_df is None):
                data_df = temp_data_df[['id', 'train_start_date', 'score']]
            else:
                data_df = pd.merge(
                    data_df,
                    temp_data_df[['id', 'train_start_date', 'score']],
                    on=['id', 'train_start_date'],
                    how='inner')
        real_label = get_real_labels()
        data_df = pd.merge(data_df, real_label, on=['id', 'train_start_date'], how='left')
        labels = data_df["grade"].to_numpy()
        features = data_df.drop(['grade', 'id', 'train_start_date'], axis=1).to_numpy()
        return features, labels


class RealVersusPredictedFeatureExtractor:

    def get_data(self, name):
        data_df = get_experiment_prediction_and_scores(name)
        label_df = get_real_labels()
        df = pd.merge(data_df, label_df, on=['id', 'train_start_date'], how='left')
        features = df[[
            'prediction1', 'prediction2', 'prediction3', 'prediction4', 'prediction5',
            'prediction6', 'prediction7',
            'real_value1', 'real_value2', 'real_value3', 'real_value4', 'real_value5', 'real_value6', 'real_value7']] \
            .to_numpy()
        features = center_and_normalize_rows(features, axis=1)
        labels = df['grade'].to_numpy()
        scores = df['score'].to_numpy()
        return features, labels, scores


class DeviationFeatureExtractor:

    def get_data(self, name):
        data_df = get_experiment_prediction_and_scores(name)
        label_df = get_real_labels()
        df = pd.merge(data_df, label_df, on=['id', 'train_start_date'], how='left')

        df["dev1"] = np.abs((df["prediction1"] - df["real_value1"])/(df["prediction1"] + 1))
        df["dev2"] = np.abs((df["prediction2"] - df["real_value2"])/(df["prediction2"] + 1))
        df["dev3"] = np.abs((df["prediction3"] - df["real_value3"])/(df["prediction3"] + 1))
        df["dev4"] = np.abs((df["prediction4"] - df["real_value4"])/(df["prediction4"] + 1))
        df["dev5"] = np.abs((df["prediction5"] - df["real_value5"])/(df["prediction5"] + 1))
        df["dev6"] = np.abs((df["prediction6"] - df["real_value6"])/(df["prediction6"] + 1))
        df["dev7"] = np.abs((df["prediction7"] - df["real_value7"])/(df["prediction7"] + 1))

        features = df[['dev1', 'dev2', 'dev3', 'dev4', 'dev5', 'dev6', 'dev7']].to_numpy()
        labels = df['grade'].to_numpy()
        scores = df['score'].to_numpy()
        return features, labels, scores


class User:
    def __init__(self, y_train):
        self.y_train = y_train

    def get_labels(self, next_indices_to_label):
        return self.y_train[next_indices_to_label]


class FullRetrainingActiveGaussianProcess():
    def __init__(self):
        self.X_known = None
        self.y_known = None
        # TODO: Is this needed?
        # kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
        # self.model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
        self.model = GaussianProcessRegressor()

    def get_name(self):
        return "gaussian-process"

    def _retrain_model(self):
        self.model.fit(self.X_known, self.y_known)

    def initial_training_of_model(self, X_new, y_new):
        self.X_known = X_new if self.X_known is None else np.vstack((self.X_known, X_new))
        self.y_known = y_new if self.y_known is None else np.append(self.y_known, y_new)
        self._retrain_model()

    def improve_model(self, X_new, y_new):
        self.X_known = X_new if self.X_known is None else np.vstack((self.X_known, X_new))
        self.y_known = y_new if self.y_known is None else np.append(self.y_known, y_new)
        self._retrain_model()

    def predict(self, X):
        y_pred = self.model.predict(X)
        return y_pred

    def get_variance(self, X):
        _, y_pred_variance = self.model.predict(X, return_std=True)
        return y_pred_variance


class FullRetrainingActiveLinearRegression():
    def __init__(self):
        self.X_known = None
        self.y_known = None
        self.model = Lasso(alpha=0.0001, max_iter=10000, positive=True)

    def get_name(self):
        return "linear-regression"

    def _retrain_model(self):
        self.model.fit(self.X_known, self.y_known)
        print(self.model.coef_)

    def initial_training_of_model(self, X_new, y_new):
        self.X_known = X_new if self.X_known is None else np.vstack((self.X_known, X_new))
        self.y_known = y_new if self.y_known is None else np.append(self.y_known, y_new)
        self._retrain_model()

    def improve_model(self, X_new, y_new):
        self.X_known = X_new if self.X_known is None else np.vstack((self.X_known, X_new))
        self.y_known = y_new if self.y_known is None else np.append(self.y_known, y_new)
        self._retrain_model()

    def predict(self, X):
        y_pred = self.model.predict(X)
        return y_pred


class FullRetrainingCommiteeActiveLinearRegression():
    def __init__(self, number_of_features):
        self.X_known = None
        self.y_known = None
        self.number_of_features = number_of_features
        self.number_of_models = 10
        self.models = [
            (random.sample(range(self.number_of_features), self.number_of_features - 2),
             Lasso(alpha=0.0001, max_iter=10000, positive=True))
            for _ in range(self.number_of_models)]

    def get_name(self):
        return "commitee-linear-regression"

    def _retrain_models(self):
        for idxs, model in self.models:
            model.fit(self.X_known[:, idxs], self.y_known)

    def initial_training_of_model(self, X_new, y_new):
        self.X_known = X_new if self.X_known is None else np.vstack((self.X_known, X_new))
        self.y_known = y_new if self.y_known is None else np.append(self.y_known, y_new)
        self._retrain_models()

    def improve_model(self, X_new, y_new):
        self.X_known = X_new if self.X_known is None else np.vstack((self.X_known, X_new))
        self.y_known = y_new if self.y_known is None else np.append(self.y_known, y_new)
        self._retrain_models()

    def predict(self, X):
        y_pred_all_models = []
        for idxs, model in self.models:
            y_pred_single_model = model.predict(X[:, idxs])
            y_pred_all_models.append(y_pred_single_model)
        y_pred_all_models = np.array(y_pred_all_models)
        y_pred = np.mean(y_pred_all_models, axis=0)
        return y_pred

    def get_variance(self, X):
        y_pred_all_models = []
        for idxs, model in self.models:
            y_pred_single_model = model.predict(X[:, idxs])
            y_pred_all_models.append(y_pred_single_model)
        y_pred_all_models = np.array(y_pred_all_models)
        y_pred_variance = np.std(y_pred_all_models, axis=0)
        return y_pred_variance


class TopDownSampleSelector(object):
    def __init__(self, model, X_train, batch_size):
        self.model = model
        self.batch_size = batch_size
        self.X_train_not_labelled = X_train
        self.is_first_sample = True

    def get_name(self):
        return "high-score-first"

    def get_next_indices_to_label(self):
        if (self.is_first_sample):
            number_of_training_samples = self.X_train_not_labelled.shape[0]
            indices = random.sample(range(number_of_training_samples), self.batch_size)
            self.is_first_sample = False
        else:
            y_pred = self.model.predict(self.X_train_not_labelled)
            indices = np.argsort(y_pred)[-self.batch_size:]
            # TODO: Remove labelled
        self.X_train_not_labelled = np.delete(self.X_train_not_labelled, indices, axis=0)
        return indices


class MostVarianceSampleSelector(object):
    def __init__(self, model, X_train, batch_size):
        self.model = model
        self.batch_size = batch_size
        self.X_train_not_labelled = X_train
        self.is_first_sample = True

    def get_name(self):
        return "high-variance-first"

    def get_next_indices_to_label(self):
        if (self.is_first_sample):
            number_of_training_samples = self.X_train_not_labelled.shape[0]
            indices = random.sample(range(number_of_training_samples), self.batch_size)
            self.is_first_sample = False
        else:
            y_pred_variance = self.model.get_variance(self.X_train_not_labelled)
            indices = np.argsort(y_pred_variance)[-self.batch_size:]
        self.X_train_not_labelled = np.delete(self.X_train_not_labelled, indices, axis=0)
        return indices


class RandomSampleSelecter(object):
    def __init__(self, X_train, batch_size):
        self.batch_size = batch_size
        self.X_train = X_train

    def get_name(self):
        return "random"

    def get_next_indices_to_label(self):
        number_of_training_samples = self.X_train.shape[0]
        indices = random.sample(range(number_of_training_samples), self.batch_size)
        return indices


class ALExperiment(object):
    def __init__(self, user, model, sample_selector, evaluator, iterations, X_train, X_test, y_train, y_test):
        self.iterations = iterations
        self.y_train = y_train
        self.X_test = X_test
        self.X_train = X_train
        self.y_test = y_test
        self.user = user
        self.model = model
        self.sample_selector = sample_selector
        self.iteration_evaluation = []
        self.iteration = 0
        self.evaluator = evaluator

    def get_name(self):
        return "AL_" + self.model.get_name() + "_" + self.sample_selector.get_name()

    def get_iteration_evaluation(self):
        return self.iteration_evaluation

    def _stop_criterie_fullfilled(self):
        self.iteration += 1
        if (self.iteration > self.iterations):
            return True
        else:
            return False

    def run(self):
        while (not self._stop_criterie_fullfilled()):
            next_indices_to_label = self.sample_selector.get_next_indices_to_label()
            new_labels = self.user.get_labels(next_indices_to_label)
            newly_labelled_features = self.X_train[next_indices_to_label]
            self.model.improve_model(newly_labelled_features, new_labels)
            y_test_pred = self.model.predict(self.X_test)
            evaluation_metric = self.evaluator.evaluate_model(y_test_pred, self.y_test)
            self.iteration_evaluation.append(evaluation_metric)


class SumOfTopEvaluator():
    def evaluate_model(self, scores, y):
        sorted_indices = np.argsort(scores)
        top_10_indices = sorted_indices[30:]
        sum_of_top_10_grades = np.sum(y[top_10_indices])
        return sum_of_top_10_grades


class RankCorrelationEvaluator():
    def evaluate_model(self, scores, y):
        return kendalltau(scores, y)[0]


def center_and_normalize_rows(x, axis=None):
    return (x - x.mean(axis, keepdims=True)) / x.mean(axis, keepdims=True)


def run_all_experiments(X, y, scores, title, is_score_representation, draw_linear_regression_seperately=False):
    experiment_evaluation_map = {}

    for _ in range(40):
        test_size = 0.30
        n = len(X)
        test_sample_size = round(n * test_size)
        all_indices = range(n)
        test_indices = np.random.choice(all_indices, size=test_sample_size, replace=False)
        train_indices = np.array(list(set(all_indices) - set(test_indices)))

        if (is_score_representation):
            X_train, X_test, y_train, y_test = \
                X[train_indices], X[test_indices], \
                y[train_indices], y[test_indices]

        else:
            X_train, X_test, y_train, y_test, score_train, score_test = \
                X[train_indices], X[test_indices], \
                y[train_indices], y[test_indices], \
                scores[train_indices], scores[test_indices]

        evaluator = RankCorrelationEvaluator()
        batch_size = 1
        iterations = int((200 * (1 - test_size)) / batch_size)
        experiments = []
        model = FullRetrainingActiveLinearRegression()
        experiment = ALExperiment(
            user=User(y_train),
            model=model,
            evaluator=evaluator,
            sample_selector=TopDownSampleSelector(model=model, X_train=X_train, batch_size=batch_size),
            iterations=iterations,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test)
        experiments.append(experiment)

        experiment = ALExperiment(
            user=User(y_train),
            model=FullRetrainingActiveLinearRegression(),
            evaluator=evaluator,
            sample_selector=RandomSampleSelecter(X_train=X_train, batch_size=batch_size),
            iterations=iterations,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test)
        experiments.append(experiment)

        model = FullRetrainingCommiteeActiveLinearRegression(X_train.shape[1])
        experiment = ALExperiment(
            user=User(y_train),
            model=model,
            evaluator=evaluator,
            sample_selector=MostVarianceSampleSelector(model=model, X_train=X_train, batch_size=batch_size),
            iterations=iterations,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test)
        experiments.append(experiment)

        model = FullRetrainingActiveGaussianProcess()
        experiment = ALExperiment(
            user=User(y_train),
            model=model,
            evaluator=evaluator,
            sample_selector=RandomSampleSelecter(X_train=X_train, batch_size=batch_size),
            iterations=iterations,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test)
        experiments.append(experiment)

        model = FullRetrainingActiveGaussianProcess()
        experiment = ALExperiment(
            user=User(y_train),
            model=model,
            evaluator=evaluator,
            sample_selector=TopDownSampleSelector(model=model, X_train=X_train, batch_size=batch_size),
            iterations=iterations,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test)
        experiments.append(experiment)

        model = FullRetrainingActiveGaussianProcess()
        experiment = ALExperiment(
            user=User(y_train),
            model=model,
            evaluator=evaluator,
            sample_selector=MostVarianceSampleSelector(model=model, X_train=X_train, batch_size=batch_size),
            iterations=iterations,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test)
        experiments.append(experiment)

        if (is_score_representation):
            for idx, score in enumerate(X_test.T):
                evaluation_of_outlier_method = evaluator.evaluate_model(score, y_test)
                outlier_plot_data = [evaluation_of_outlier_method for _ in range(iterations)]
                name = "Score" + str(idx)
                if (name not in experiment_evaluation_map):
                    experiment_evaluation_map[name] = []
                experiment_evaluation_map[name].append(outlier_plot_data)
        else:
            evaluation_of_outlier_method = evaluator.evaluate_model(score_test, y_test)
            outlier_plot_data = [evaluation_of_outlier_method for _ in range(iterations)]

            name = "Score"
            if (name not in experiment_evaluation_map):
                experiment_evaluation_map[name] = []
            experiment_evaluation_map[name].append(outlier_plot_data)

        for experiment in experiments:
            experiment.run()
            evaluations = experiment.get_iteration_evaluation()
            name = experiment.get_name()
            if (name not in experiment_evaluation_map):
                experiment_evaluation_map[name] = []
            experiment_evaluation_map[name].append(evaluations)
    for name in experiment_evaluation_map:
        experiment_evaluations = np.array(experiment_evaluation_map[name])
        experiment_evaluation_means = np.mean(experiment_evaluations, axis=0)
        plt.plot(experiment_evaluation_means, label=name)

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='xx-small')
    plt.title(title)
    plt.show()

    if(draw_linear_regression_seperately):
        for name in experiment_evaluation_map:
            if('gauss' in name):
                continue
            experiment_evaluations = np.array(experiment_evaluation_map[name])
            experiment_evaluation_means = np.mean(experiment_evaluations, axis=0)
            plt.plot(experiment_evaluation_means, label=name)

        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='xx-small')
        plt.title(title + '(linear methods only)')
        plt.show()


grades = get_real_labels()['grade']
bins = np.arange(0, grades.max() + 1.5) - 0.5
plt.hist(grades, bins)
plt.title("Outlier Grade Distribution")
plt.show()

title = "AL on Deviation from Prediction"
X, y, scores = DeviationFeatureExtractor() \
    .get_data("iceberg-True_drop-True_dense-16_lstm-8_training-sample-15000_epocs-1000_min-window-10000")
run_all_experiments(X, y, scores, title, False)

title = "AL on Prediction and Real Observations"
X, y, scores = RealVersusPredictedFeatureExtractor() \
    .get_data("iceberg-True_drop-True_dense-16_lstm-8_training-sample-15000_epocs-1000_min-window-10000")
run_all_experiments(X, y, scores, title, False)

title = "AL on Outlier Score Representation"
X, y = OutlieScoreFeatureExtractor() \
    .get_data("iceberg-True_drop-True_dense-16_lstm-8_training-sample-15000_epocs-1000_min-window-10000")
run_all_experiments(X, y, scores, title, True, True)

from cProfile import label
from itertools import cycle

from sklearn.linear_model import SGDClassifier
from library_python_script.tools.predictor import DNN_Predictor, Linear_Regression, RF_Predictor, DNN_auto_Classifier, TSNE_Visualisator, UMAP_Visualisator, SVM_Predictor, Gaussian_Process, Gradient_boosting
import tensorflow as tf
from library_python_script.tools.metrics import r2
import visualkeras
import os
import matplotlib.pyplot as plt
from tensorflow import keras
from library_python_script.tools.plotLearning import PlotLearning
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
import numpy as np
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, Matern, DotProduct
from sklearn.metrics import auc, confusion_matrix, roc_curve
from sklearn.metrics import precision_score, accuracy_score, recall_score
from sklearn.metrics import precision_recall_fscore_support
from scipy import interp
from sklearn.metrics import RocCurveDisplay, roc_auc_score

from library_python_script.tools.utility import OneHotEncoding



class DNN_Classifier(DNN_Predictor):
    def __init__(self, data=None, json_models=None, metrics = ['mse', 'accuracy', 'recall', 'precision'], epoch=1000, iteration=5, live=True, early_stopping = [True, 300], batch_size=10, verbose=0):
        lr_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=1e-2,
            decay_steps=10000,
            decay_rate=0.9)
        self.optimizer = "adam" #keras.optimizers.Adam(learning_rate=lr_schedule)
        self.epoch = epoch
        self.batch_size = batch_size
        self.loss = "categorical_crossentropy"
        self.metrics = ["mse", tf.keras.metrics.Accuracy(), tf.keras.metrics.Recall(), tf.keras.metrics.Precision()]
        self.metrics_str = metrics
        self.verbose = verbose
        self.data = data
        self.iteration = iteration
        self.callback = []
        if early_stopping[0]:
            self.callback.append(tf.keras.callbacks.EarlyStopping(monitor='loss', patience=early_stopping[1]))
        if live:
            self.callback.append(PlotLearning())
        self.history_memory = []
        json_models = self.set_model_info(json_models)
        self.model_info = json_models
    
    def keep_history_memory(self):
        super().keep_history_memory()
        self.prediction.append(self.model.predict(self.data.get_x_test()))
    
    def run(self, save_path=None):
        axis_full = [*self.metrics_str, *["val_" + i for i in self.metrics_str]]
        self.history_memory = {axis :{"history" : [], "means" : [], "min" : [], "max" : []} for axis in axis_full}
        self.prediction = []
        for it in range(0, self.iteration):
            self.build_model_from_dict(self.model_info)
            self.c_history = self.model.fit(self.data.get_x_train(),
                        self.data.get_y_train(ravel=False, onehot=True),
                        epochs          = self.epoch,
                        batch_size      = self.batch_size,
                        verbose         = 0, # self.verbose,
                        validation_data = (self.data.get_x_test(), self.data.get_y_test(ravel=False, onehot=True)),
                        callbacks = [self.callback])
            self.keep_history_memory()
            self.it_order = np.argsort(self.history_memory["val_accuracy"]['max'])[::-1] # ORDER ON accuracy VALUES !
            if save_path!=None:
                self.save_best_model(save_path)
            visualkeras.layered_view(self.model, legend=True, to_file=save_path + 'model/model.png')

    def runGridSearch(self, batch_size=[10, 20, 40, 60, 80, 100, 150, 200], epochs = [100, 200, 300, 400, 500, 1000]):
        self.build_model_from_dict(self.model_info)
        param_grid = dict(batch_size=batch_size, epochs=epochs)
        build_model = lambda: self.model
        model = KerasClassifier(build_fn=build_model, verbose=0)
        grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1)
        grid_result = grid.fit(self.data.get_x_train(),
                    self.data.get_y_train(ravel=False, onehot=True))
        print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    
    def plot(self, save_path=None, figsize=(16, 12), save_only=False):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if save_only:
            file_id = 0
            while os.path.isfile(save_path + str(file_id) + '_' + "accuracy" +  ".png"):
                file_id += 1

        prediction = self.prediction[self.it_order[-1]]
        matrix = confusion_matrix(np.argmax(prediction, axis=1), np.argmax(self.data.get_y_test(ravel=False, onehot=True), axis=1), labels=self.data.nb_class)
        fig, ax = plt.subplots()
        cax = ax.matshow(matrix)
        fig.colorbar(cax)
        ax.set_xlabel('True Label')
        ax.set_ylabel('Predicted Label')
        for (i, j), z in np.ndenumerate(matrix):
            ax.text(j, i, '{:0.1f}'.format(z), ha='center', va='center')
        
        if save_only:
            plt.savefig(save_path + str(file_id) + "_" + "accuracy" + ".png")
        else:
            plt.show()
        plt.close('all')

        alpha_color = [0.4]*len(self.it_order)
        alpha_color[-1] = 1.0
        plt.figure(figsize=figsize)
        plt.ylabel("Loss")
        plt.xlabel('Epoch')
        for it_id in range(0, len(self.it_order)):
            plt.plot(self.history_memory['mse']['history'][self.it_order[it_id]], color="tab:blue", alpha = alpha_color[it_id])
            plt.plot(self.history_memory["val_mse"]['history'][self.it_order[it_id]], color="tab:orange", alpha = alpha_color[it_id])
        plt.legend(["mse", "val_mse"], loc='upper left')
        x1,x2,y1,y2 = plt.axis()  
        plt.axis((x1,x2,0.0,1.0))
        if save_only:
                plt.savefig(save_path + str(file_id) + "_mse_fit" + ".png")
        else:
            plt.show()
        plt.close('all')

        plt.figure(figsize=figsize)
        plt.ylabel("Accuracy")
        plt.xlabel('Epoch')
        for it_id in range(0, len(self.it_order)):
            plt.plot(self.history_memory['accuracy']['history'][self.it_order[it_id]], color="tab:blue", alpha = alpha_color[it_id])
            plt.plot(self.history_memory["val_accuracy"]['history'][self.it_order[it_id]], color="tab:orange", alpha = alpha_color[it_id])
        plt.legend(["accuracy", "val_accuracy"], loc='upper left')
        x1,x2,y1,y2 = plt.axis()  
        plt.axis((x1,x2,0.0,1.0))
        if save_only:
                plt.savefig(save_path + str(file_id) + "_accuracy_fit" + ".png")
        else:
            plt.show()
        plt.close('all')

        lw = 2
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in self.data.nb_class:
            fpr[i], tpr[i], _ = roc_curve(self.data.get_y_test(ravel=False, onehot=True)[:, i], prediction[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        fpr["micro"], tpr["micro"], _ = roc_curve(self.data.get_y_test(ravel=True, onehot=True), prediction.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        all_fpr = np.unique(np.concatenate([fpr[i] for i in self.data.nb_class]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in self.data.nb_class:
            mean_tpr += interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= len(self.data.nb_class)
        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

        plt.plot(fpr["macro"], tpr["macro"],
                label='macro-average ROC curve (area = {0:0.2f})'
                    ''.format(roc_auc["macro"]),
                color='navy', linestyle=':', linewidth=4)

        colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'darkgreen', 'lightpink'])
        for i, color in zip(self.data.nb_class, colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                    label='ROC curve of class {0} (area = {1:0.2f})'
                    ''.format(i, roc_auc[i]))

        plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc="lower right")
        if save_only:
            plt.savefig(save_path + str(file_id) + "_" + "precision_recall" + ".png")
        else:
            plt.show()
        self.score = {}
        with open(save_path + str(file_id) + ".txt", 'w') as file:
            self.score['precision'], self.score['recall'], self.score['fscore'], self.score['support'] = precision_recall_fscore_support(np.array([str(x) for x in self.data.get_y_test(onehot=False)]), np.array([str(x) for x in np.argmax(prediction, axis=1)]), average=None, labels = self.data.nb_class)
            self.score["accuracy"] = accuracy_score(self.data.get_y_test(onehot=False), np.argmax(prediction, axis=1))
            for metrics in ['accuracy', "precision", "recall", "fscore", "support"]:
                file.write(metrics + ' : ' + str(self.score[metrics]) + "\n")

class RF_Classifier(RF_Predictor):
    def run(self):
        self.model = RandomForestClassifier(max_depth=None, random_state=0)
        self.model.fit(self.data.get_x_train(), self.data.get_y_train())
        self.pred = self.model.predict(self.data.get_x_test())
    
    def init_metrics(self):
        self.metrics = [accuracy_score, None, None, None, None]
        self.metrics_str = ["accuracy", "precision", "recall", 'fscore', 'support']
    
    def plot(self, save_path=None, save_only=True):
        self.create_resdir(save_path, save_only)
        prediction = self.model.predict_proba(self.data.get_x_test())
        matrix = confusion_matrix(np.argmax(prediction, axis=1), self.data.get_y_test(ravel=False), labels=self.data.nb_class)
        fig, ax = plt.subplots()
        cax = ax.matshow(matrix)
        fig.colorbar(cax)
        ax.set_xlabel('True Label')
        ax.set_ylabel('Predicted Label')
        for (i, j), z in np.ndenumerate(matrix):
            ax.text(j, i, '{:0.1f}'.format(z), ha='center', va='center')
        
        if save_only:
            plt.savefig(save_path + str(self.file_id) + "_" + "accuracy" + ".png")
        else:
            plt.show()
        plt.close('all')
        
        # PLOT ROC AUC CURVE
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        y_test_onehot = OneHotEncoding(self.data.get_y_test(), self.data.nb_class)
        pred_onehot = prediction
        for i in self.data.nb_class:
            fpr[i], tpr[i], _ = roc_curve(y_test_onehot[:, i], pred_onehot[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test_onehot.ravel(), pred_onehot.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        all_fpr = np.unique(np.concatenate([fpr[i] for i in self.data.nb_class]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in self.data.nb_class:
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= len(self.data.nb_class)
        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
        plt.figure()
        plt.plot(
            fpr["micro"],
            tpr["micro"],
            label="micro-average ROC curve (area = {0:0.2f})".format(roc_auc["micro"]),
            color="deeppink",
            linestyle=":",
            linewidth=4,
        )

        plt.plot(
            fpr["macro"],
            tpr["macro"],
            label="macro-average ROC curve (area = {0:0.2f})".format(roc_auc["macro"]),
            color="navy",
            linestyle=":",
            linewidth=4,
        )

        colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'darkgreen', 'lightpink'])
        for i, color in zip(self.data.nb_class, colors):
            plt.plot(
                fpr[i],
                tpr[i],
                color=color,
                lw=2,
                label="ROC curve of class {0} (area = {1:0.2f})".format(i, roc_auc[i]),
            )

        plt.plot([0, 1], [0, 1], "k--", lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc="lower right")
        if save_only:
            plt.savefig(save_path + str(self.file_id) + "_" + "ROC" + ".png")
        else:
            plt.show()
        
        self.score = {}
        with open(save_path + str(self.file_id) + ".txt", 'w') as file:
            self.score['precision'], self.score['recall'], self.score['fscore'], self.score['support'] = precision_recall_fscore_support(np.array([str(x) for x in self.data.get_y_test()]), np.array([str(x) for x in self.pred]), average=None, labels = self.data.nb_class)
            for i_metrics in range(0, len(self.metrics)):
                if(self.metrics_str[i_metrics]=="accuracy"):
                    self.score[self.metrics_str[i_metrics]] = self.metrics[i_metrics](self.data.get_y_test(), self.pred)
                else:
                    pass
                    # self.score[self.metrics_str[i_metrics]] = self.metrics[i_metrics](self.data.get_y_test(), self.pred, average="micro", labels = self.data.nb_class)
                file.write(self.metrics_str[i_metrics] + ' : ' + str(self.score[self.metrics_str[i_metrics]]) + "\n")
        
class TSNE_Classifier(TSNE_Visualisator):
    pass

class UMAP_Classifier(UMAP_Visualisator):
    pass

class SVM_Classifier(SVM_Predictor):
    def __init__(self, data=None):
        super().__init__(data)
        self.init_metrics()

    def build_model_from_dict(self, model_dict={'dfs':'ovo'}):
        self.model = svm.SVC(probability=True)
    
    def init_metrics(self):
        self.metrics = [accuracy_score, None, None, None, None]
        self.metrics_str = ["accuracy", "precision", "recall", 'fscore', 'support']
    
    def create_resdir(self, save_path, save_only):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if save_only:
            self.file_id = 0
            while os.path.isfile(save_path + str(self.file_id) +  ".txt"):
                self.file_id += 1
    def run(self):
        super().run()
        self.pred = self.model.predict(self.data.get_x_test())
        self.decision_function = self.model.decision_function(self.data.get_x_test())

    def plot(self, save_path=None, save_only=True):
        self.create_resdir(save_path, save_only)
        prediction = self.model.predict_proba(self.data.get_x_test())
        matrix = confusion_matrix(np.argmax(prediction, axis=1), self.data.get_y_test(ravel=False), labels=self.data.nb_class)
        fig, ax = plt.subplots()
        cax = ax.matshow(matrix)
        fig.colorbar(cax)
        ax.set_xlabel('True Label')
        ax.set_ylabel('Predicted Label')
        for (i, j), z in np.ndenumerate(matrix):
            ax.text(j, i, '{:0.1f}'.format(z), ha='center', va='center')
        
        if save_only:
            plt.savefig(save_path + str(self.file_id) + "_" + "accuracy" + ".png")
        else:
            plt.show()
        plt.close('all')

        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        y_test_onehot = OneHotEncoding(self.data.get_y_test(), self.data.nb_class)
        pred_onehot = self.decision_function
        for i in self.data.nb_class:
            fpr[i], tpr[i], _ = roc_curve(y_test_onehot[:, i], pred_onehot[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test_onehot.ravel(), pred_onehot.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        all_fpr = np.unique(np.concatenate([fpr[i] for i in self.data.nb_class]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in self.data.nb_class:
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= len(self.data.nb_class)
        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
        plt.figure()
        plt.plot(
            fpr["micro"],
            tpr["micro"],
            label="micro-average ROC curve (area = {0:0.2f})".format(roc_auc["micro"]),
            color="deeppink",
            linestyle=":",
            linewidth=4,
        )

        plt.plot(
            fpr["macro"],
            tpr["macro"],
            label="macro-average ROC curve (area = {0:0.2f})".format(roc_auc["macro"]),
            color="navy",
            linestyle=":",
            linewidth=4,
        )

        colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'darkgreen', 'lightpink'])
        for i, color in zip(self.data.nb_class, colors):
            plt.plot(
                fpr[i],
                tpr[i],
                color=color,
                lw=2,
                label="ROC curve of class {0} (area = {1:0.2f})".format(i, roc_auc[i]),
            )

        plt.plot([0, 1], [0, 1], "k--", lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc="lower right")
        if save_only:
            plt.savefig(save_path + str(self.file_id) + "_" + "ROC" + ".png")
        else:
            plt.show()
        
        # Save metrics results
        self.score = {}
        with open(save_path + str(self.file_id) + ".txt", 'w') as file:
            self.score['precision'], self.score['recall'], self.score['fscore'], self.score['support'] = precision_recall_fscore_support(np.array([str(x) for x in self.data.get_y_test()]), np.array([str(x) for x in self.pred]), average=None, labels = self.data.nb_class)
            for i_metrics in range(0, len(self.metrics)):
                if(self.metrics_str[i_metrics]=="accuracy"):
                    self.score[self.metrics_str[i_metrics]] = self.metrics[i_metrics](self.data.get_y_test(), self.pred)
                else:
                    pass
                    # self.score[self.metrics_str[i_metrics]] = self.metrics[i_metrics](self.data.get_y_test(), self.pred, average="micro", labels = self.data.nb_class)
                file.write(self.metrics_str[i_metrics] + ' : ' + str(self.score[self.metrics_str[i_metrics]]) + "\n")

class Gaussian_Process_Classifier(Gaussian_Process):
    def __init__(self, data=None):
        super().__init__(data)
        self.init_metrics()
    
    def runGridSearch(self, param_grid = None):
        param_grid = [{"kernel": [RBF(length_scale=l, length_scale_bounds="fixed") for l in [1.0, 10.0, 100.0, 1000.0]]}]
        scores = ['explained_variance', 'r2']
        gp = GaussianProcessClassifier(n_restarts_optimizer=18, multi_class="one_vs_one")
        for score in scores:
            print("# Tuning hyper-parameters for %s" % score)
            clf = GridSearchCV(estimator=gp, param_grid=param_grid, cv=4,
                            scoring='%s' % score)
            clf.fit(self.data.get_x_train(), self.data.get_y_train())
            print("Best: %f using %s" % (clf.best_score_, clf.best_params_))

    def build_model_from_dict(self, model_dict = {"0" : {'type' : 'RBF', 'length' : 16384}, "1" : {'type' : 'MATERN', 'length' : 16384, 'nu' : 0.5}, 'OP' : '*'}):
        kernel = {}
        for i in range((len(model_dict)-1)):
            if model_dict[str(i)]['type'] == 'RBF':
                kernel[i] = RBF(length_scale=model_dict[str(i)]['length'], length_scale_bounds=(model_dict[str(i)]['length_scale_bounds'][0], model_dict[str(i)]['length_scale_bounds'][1])  if 'length_scale_bounds' in model_dict[str(i)].keys() else "fixed")
            if model_dict[str(i)]['type'] == 'MATERN':
                kernel[i] = Matern(nu=model_dict[str(i)]['nu'], length_scale=model_dict[str(i)]['length'], length_scale_bounds="fixed")
            if model_dict[str(i)]['type'] == 'FLOAT':
                kernel[i] = model_dict[str(i)]['value']
        if model_dict['OP'] == '+':
            self.kernel = kernel[0] + kernel[1]
        elif model_dict['OP'] == '-':
            self.kernel = kernel[0] - kernel[1]
        elif model_dict['OP'] == '*':
            self.kernel = kernel[0] * kernel[1]
        else:
            pass
        if len(kernel)==0:
            self.model = GaussianProcessClassifier(kernel=self.kernel, n_restarts_optimizer=18, multi_class="one_vs_one")
        else:
            self.model = GaussianProcessClassifier(kernel=self.kernel, n_restarts_optimizer=18, multi_class="one_vs_one")
    
    def init_metrics(self):
        self.metrics = [accuracy_score, None, None, None, None]
        self.metrics_str = ["accuracy", "precision", "recall", 'fscore', 'support']
    
    def run(self, save_path=None):
        self.model.fit(self.data.get_x_train(), self.data.get_y_train())
        # self.pred = self.model.predict(self.data.get_x_test().to_numpy())
        # self.pred = self.pred.flatten()

    def plot(self, save_path=None, save_only=False):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if save_only:
            file_id = 0
            while os.path.isfile(save_path + str(file_id) +  ".txt"):
                file_id += 1
        prediction = self.model.predict(self.data.get_x_test())
        matrix = confusion_matrix(prediction.ravel(), self.data.get_y_test(ravel=True), labels=self.data.nb_class)
        fig, ax = plt.subplots()
        cax = ax.matshow(matrix)
        fig.colorbar(cax)
        ax.set_xlabel('True Label')
        ax.set_ylabel('Predicted Label')
        for (i, j), z in np.ndenumerate(matrix):
            ax.text(j, i, '{:0.1f}'.format(z), ha='center', va='center')
        
        if save_only:
            plt.savefig(save_path + str(file_id) + "_" + "accuracy" + ".png")
        else:
            plt.show()
        plt.close('all')        
        # Save metrics results
        self.score = {}
        with open(save_path + str(file_id) + ".txt", 'w') as file:
            self.score['precision'], self.score['recall'], self.score['fscore'], self.score['support'] = precision_recall_fscore_support(np.array([str(x) for x in self.data.get_y_test()]), np.array([str(x) for x in prediction]), average=None, labels = self.data.nb_class)
            self.score['accuracy'] = accuracy_score(self.data.get_y_test(), prediction)
            for i_metrics in range(0, len(self.metrics)):
                file.write(self.metrics_str[i_metrics] + ' : ' + str(self.score[self.metrics_str[i_metrics]]) + "\n")

class Gradient_boosting_Classifier(Gradient_boosting):
    def __init__(self, data=None):
        super().__init__(data)
        self.init_metrics()

    def build_model_from_dict(self, model_dict = {"random_state" : 123}):
        self.model = GradientBoostingClassifier(random_state=model_dict['random_state'])

    def init_metrics(self):
        self.metrics = [accuracy_score, None, None, None, None]
        self.metrics_str = ["accuracy", "precision", "recall", 'fscore', 'support']
    
    def run(self, save_path=None):
        self.model.fit(self.data.get_x_train(), self.data.get_y_train())
        self.pred = self.model.predict(self.data.get_x_test())

    def plot(self, save_path=None, figsize=..., save_only=False):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if save_only:
            file_id = 0
            while os.path.isfile(save_path + str(file_id) +  ".txt"):
                file_id += 1
        # plot_confusion_matrix(self.model, self.data.get_x_test().to_numpy(), self.data.get_y_test())
        matrix = confusion_matrix(self.pred.ravel(), self.data.get_y_test(ravel=True), labels=self.data.nb_class)
        fig, ax = plt.subplots()
        cax = ax.matshow(matrix)
        fig.colorbar(cax)
        ax.set_xlabel('True Label')
        ax.set_ylabel('Predicted Label')
        for (i, j), z in np.ndenumerate(matrix):
            ax.text(j, i, '{:0.1f}'.format(z), ha='center', va='center')
        
        if save_only:
            plt.savefig(save_path + str(file_id) + "_" + "accuracy" + ".png")
        else:
            plt.show()
        plt.close('all')
        # self.accuracy = {}
        # with open(save_path + str(file_id) + ".txt", 'w') as file:
        #     for i_metrics in range(0, len(self.metrics)):
        #         if(self.metrics_str[i_metrics]=="accuracy"):
        #             self.accuracy[self.metrics_str[i_metrics]] = self.metrics[i_metrics](self.data.get_y_test(), self.pred)
        #         else:
        #             self.accuracy[self.metrics_str[i_metrics]] = self.metrics[i_metrics](self.data.get_y_test(), self.pred, average="micro")
        #         file.write(self.metrics_str[i_metrics] + ' : ' + str(self.accuracy[self.metrics_str[i_metrics]]) + "\n")
        self.score = {}
        with open(save_path + str(file_id) + ".txt", 'w') as file:
            self.score['precision'], self.score['recall'], self.score['fscore'], self.score['support'] = precision_recall_fscore_support(np.array([str(x) for x in self.data.get_y_test()]), np.array([str(x) for x in self.pred]), average=None, labels = self.data.nb_class)
            self.score['accuracy'] = accuracy_score(self.data.get_y_test(), self.pred)
            for i_metrics in range(0, len(self.metrics)):
                file.write(self.metrics_str[i_metrics] + ' : ' + str(self.score[self.metrics_str[i_metrics]]) + "\n")

class Linear_Classifier_SGD(Linear_Regression):

    def __init__(self, data=None):
        super().__init__(data)
        self.init_metrics()

    def init_metrics(self):
        self.metrics = [accuracy_score, None, None, None, None]
        self.metrics_str = ["accuracy", "precision", "recall", 'fscore', 'support']

    def build_model_from_dict(self, model_dict=...):
        penalty = 'l2'
        self.method = "Standard"
        if 'method' in model_dict.keys():
            method_penalty = {'Standard' : 'l2', 'Lasso' : 'l1', 'elasticnet' : 'Elasticnet'}
            self.model = SGDClassifier(max_iter=1000, tol=1e-3, penalty=method_penalty[model_dict['method']])
            self.method = model_dict["method"]
        self.model = SGDClassifier(max_iter=1000, tol=1e-3)
        
    
    def run(self, save_path=None):
        self.model.fit(self.data.get_x_train(), self.data.get_y_train(ravel=True))
        self.pred = self.model.predict(self.data.get_x_test())
        self.decision_function = self.model.decision_function(self.data.get_x_test())

    def create_resdir(self, save_path, save_only):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if save_only:
            self.file_id = 0
            while os.path.isfile(save_path + str(self.file_id) +  ".txt"):
                self.file_id += 1
                
    def plot(self, save_path=None, save_only=True):
        save_path = save_path
        self.create_resdir(save_path, save_only)
        prediction = self.decision_function#self.model.predict_proba(self.data.get_x_test().to_numpy())
        matrix = confusion_matrix(np.argmax(prediction, axis=1), self.data.get_y_test(ravel=False), labels=self.data.nb_class)
        fig, ax = plt.subplots()
        cax = ax.matshow(matrix)
        fig.colorbar(cax)
        ax.set_xlabel('True Label')
        ax.set_ylabel('Predicted Label')
        for (i, j), z in np.ndenumerate(matrix):
            ax.text(j, i, '{:0.1f}'.format(z), ha='center', va='center')
        
        if save_only:
            plt.savefig(save_path + str(self.file_id) + "_" + "accuracy" + ".png")
        else:
            plt.show()
        plt.close('all')
        # plot_confusion_matrix(self.model, self.data.get_x_test().to_numpy(), self.data.get_y_test())
        self.score = {}
        with open(save_path + str(self.file_id) + ".txt", 'w') as file:
            self.score['precision'], self.score['recall'], self.score['fscore'], self.score['support'] = precision_recall_fscore_support(np.array([str(x) for x in self.data.get_y_test()]), np.array([str(x) for x in self.pred]), average=None, labels = self.data.nb_class)
            for i_metrics in range(0, len(self.metrics)):
                if(self.metrics_str[i_metrics]=="accuracy"):
                    self.score[self.metrics_str[i_metrics]] = self.metrics[i_metrics](self.data.get_y_test(), self.pred)
                file.write(self.metrics_str[i_metrics] + ' : ' + str(self.score[self.metrics_str[i_metrics]]) + "\n")
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        y_test_onehot = OneHotEncoding(self.data.get_y_test(), self.data.nb_class)
        pred_onehot = self.decision_function
        for i in self.data.nb_class:
            fpr[i], tpr[i], _ = roc_curve(y_test_onehot[:, i], pred_onehot[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test_onehot.ravel(), pred_onehot.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        all_fpr = np.unique(np.concatenate([fpr[i] for i in self.data.nb_class]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in self.data.nb_class:
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= len(self.data.nb_class)
        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
        plt.figure()
        plt.plot(
            fpr["micro"],
            tpr["micro"],
            label="micro-average ROC curve (area = {0:0.2f})".format(roc_auc["micro"]),
            color="deeppink",
            linestyle=":",
            linewidth=4,
        )

        plt.plot(
            fpr["macro"],
            tpr["macro"],
            label="macro-average ROC curve (area = {0:0.2f})".format(roc_auc["macro"]),
            color="navy",
            linestyle=":",
            linewidth=4,
        )

        colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'darkgreen', 'lightpink'])
        for i, color in zip(self.data.nb_class, colors):
            plt.plot(
                fpr[i],
                tpr[i],
                color=color,
                lw=2,
                label="ROC curve of class {0} (area = {1:0.2f})".format(i, roc_auc[i]),
            )

        plt.plot([0, 1], [0, 1], "k--", lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc="lower right")
        if save_only:
            plt.savefig(save_path + str(self.file_id) + "_" + "ROC" + ".png")
        else:
            plt.show()


    






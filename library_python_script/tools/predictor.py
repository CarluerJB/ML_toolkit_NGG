from mimetypes import init
from turtle import color
import os
from matplotlib import projections
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow import keras
import tensorflow as tf
from library_python_script.tools.metrics import MSE_GP, rmse, r2, RMSE_GP, R2_GP, swish
from library_python_script.tools.plotLearning import PlotLearning
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV, ElasticNetCV
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.manifold import TSNE
import numpy as np
from tensorflow.keras.metrics import Accuracy, AUC, Precision, Recall
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
from umap import UMAP
import umap.plot 
import plotly.express as px
import pandas as pd

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR

import visualkeras
from PIL import ImageFont



from tensorflow.python.keras.utils.generic_utils import get_custom_objects
from tensorflow.keras.layers import Activation
get_custom_objects().update({'swish': Activation(swish)})


class DNN_Predictor():
    def __init__(self, data=None, json_models=None, metrics = ['r2', 'mse', 'rmse'], epoch=1000, iteration=5, live=True, early_stopping = [True, 300], batch_size=10, verbose=0):
        
        lr_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=1e-2,
            decay_steps=10000,
            decay_rate=0.9)
        self.optimizer = 'adam' #keras.optimizers.Adam(learning_rate=lr_schedule)
        self.epoch = epoch
        self.batch_size = batch_size
        self.loss = "mse"
        self.metrics = [rmse, 'mse', r2]
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
    
    def set_batchsize(self, batchsize):
        self.batch_size = batchsize
    
    def set_epoch(self, epoch):
        self.epoch = epoch

    def set_model_info(self, json_models):
        for node_id in json_models["inputs"].keys():
            if json_models["inputs"][node_id]==None:
                json_models['inputs'][node_id]=(self.data.x_train[0].shape[1], )
        for layer_id, layers in json_models['layers'].items():
            for node in range(0,len(layers)):
                if layers[node]["type"]=="Dropout":
                    json_models["layers"][layer_id][node]["size"]=(self.data.x_train[0].shape[1], )
                if (layers[node]["type"]=="Dense") and (layers[node]["size"]==None):
                    json_models["layers"][layer_id][node]["size"]=self.data.x_train[0].shape[1]
        return json_models

    def build_model_from_dict(self, model_info):
        self.model_info = model_info
        keras.backend.clear_session()
        input_list = []
        layer_list = []
        for input in model_info['inputs'].values():
            input_list.append(keras.layers.Input(input))
        layer_list.append(input_list)
        for id, layer in model_info['layers'].items():
            node_list = []
            for node in layer:
                if node['type'] == "Transfer":
                    node_list.append(self.model(layer_list[node['inputs']['layer']][node['inputs']['node']]))
                if node['type'] == "Dense":
                    node_list.append(keras.layers.Dense(node['size'], activation=node['activation'] if 'activation' in node.keys() else 'linear')(layer_list[node['inputs']['layer']][node['inputs']['node']]))
                if node['type'] == "Relu":
                    node_list.append(keras.layers.Activation(keras.activations.relu)(layer_list[node['inputs']['layer']][node['inputs']['node']]))
                if node['type'] == "Sigmoid":
                    node_list.append(keras.layers.Activation(keras.activations.sigmoid)(layer_list[node['inputs']['layer']][node['inputs']['node']]))
                if node['type'] == "Softplus":
                    node_list.append(keras.layers.Activation(keras.activations.softplus)(layer_list[node['inputs']['layer']][node['inputs']['node']]))
                if node['type'] == "Softsign":
                    node_list.append(keras.layers.Activation(keras.activations.softsign)(layer_list[node['inputs']['layer']][node['inputs']['node']]))
                if node['type'] == "Tanh":
                    node_list.append(keras.layers.Activation(keras.activations.tanh)(layer_list[node['inputs']['layer']][node['inputs']['node']]))
                if node['type'] == "Selu":
                    node_list.append(keras.layers.Activation(keras.activations.selu)(layer_list[node['inputs']['layer']][node['inputs']['node']]))
                if node['type'] == "Elu":
                    node_list.append(keras.layers.Activation(keras.activations.elu)(layer_list[node['inputs']['layer']][node['inputs']['node']]))
                if node['type'] == "Swish":
                    node_list.append(keras.layers.Activation(keras.activations.swish)(layer_list[node['inputs']['layer']][node['inputs']['node']]))
                if node['type'] == "Exponential":
                    node_list.append(keras.layers.Activation(keras.activations.exponential)(layer_list[node['inputs']['layer']][node['inputs']['node']]))
                if node['type'] == "Softmax":
                    node_list.append(keras.layers.Activation(keras.activations.softmax)(layer_list[node['inputs']['layer']][node['inputs']['node']]))
                if node['type'] == "BatchNorm":
                    node_list.append(keras.layers.BatchNormalization()(layer_list[node['inputs']['layer']][node['inputs']['node']]))
                if node['type'] == "Dropout":
                    node_list.append(keras.layers.Dropout(node['ratio'], input_shape=node['size'])(layer_list[node['inputs']['layer']][node['inputs']['node']]))
                if node['type'] == "Concat":
                    node_list.append(keras.layers.Concatenate()([layer_list[node['inputs']['layer']][i_node] for i_node in node['inputs']['node']]))
                if node['type'] == "Average":
                    node_list.append(keras.layers.Average()([layer_list[node['inputs']['layer']][i_node] for i_node in node['inputs']['node']]))
                if node['type'] == "Max":
                    node_list.append(keras.layers.Maximum()([layer_list[node['inputs']['layer']][i_node] for i_node in node['inputs']['node']]))
                if node['type'] == "Min":
                    node_list.append(keras.layers.Minimum()([layer_list[node['inputs']['layer']][i_node] for i_node in node['inputs']['node']]))
                if node['type'] == "Add":
                    node_list.append(keras.layers.Add()([layer_list[node['inputs']['layer']][i_node] for i_node in node['inputs']['node']]))
                if node['type'] == "Sub":
                    node_list.append(keras.layers.Subtract()([layer_list[node['inputs']['layer']][i_node] for i_node in node['inputs']['node']]))
                if node['type'] == "Dot":
                    node_list.append(keras.layers.Dot(axes=node["axes"], normalize=node["normalize"])([layer_list[node['inputs']['layer']][i_node] for i_node in node['inputs']['node']]))
            layer_list.append(node_list)
        self.model = keras.Model(layer_list[0], layer_list[-1])
        self.model.compile(optimizer = self.optimizer,
                    loss      = self.loss,
                    metrics   = self.metrics )

    def run(self, save_path=None):
        axis_full = [*self.metrics_str, *["val_" + i for i in self.metrics_str]]
        self.history_memory = {axis :{"history" : [], "means" : [], "min" : [], "max" : []} for axis in axis_full}
        for it in range(0, self.iteration):
            self.build_model_from_dict(self.model_info)
            if len(self.data.x_train)>1:
                if len(self.data.y_train)>1:
                    self.c_history = self.model.fit(self.data.x_train,
                        self.data.y_train,
                        epochs          = self.epoch,
                        batch_size      = self.batch_size,
                        verbose         = self.verbose,
                        validation_data = (self.data.x_test, self.data.y_test),
                        callbacks = [self.callback])
                else:
                    self.c_history = self.model.fit(self.data.x_train,
                        self.data.y_train[0],
                        epochs          = self.epoch,
                        batch_size      = self.batch_size,
                        verbose         = self.verbose,
                        validation_data = (self.data.x_test, self.data.y_test[0]),
                        callbacks = [self.callback])
            else:
                if len(self.data.y_train)>1:
                    self.c_history = self.model.fit(self.data.x_train[0],
                        self.data.y_train,
                        epochs          = self.epoch,
                        batch_size      = self.batch_size,
                        verbose         = self.verbose,
                        validation_data = (self.data.x_test[0], self.data.y_test),
                        callbacks = [self.callback])
                else:
                    self.c_history = self.model.fit(self.data.x_train[0],
                        self.data.y_train[0],
                        epochs          = self.epoch,
                        batch_size      = self.batch_size,
                        verbose         = self.verbose,
                        validation_data = (self.data.x_test[0], self.data.y_test[0]),
                        callbacks = [self.callback])
            self.keep_history_memory()
            self.it_order = np.argsort(self.history_memory["val_rmse"]['min'])[::-1] # ORDER ON RMSE VALUES !
            if save_path!=None:
                self.save_best_model(save_path)
            visualkeras.layered_view(self.model, legend=True, to_file=save_path + 'model/model.png')
    
    def runGridSearch(self, batch_size=[20, 40, 60, 80, 100, 150, 200], epochs = [200, 1000]):
        self.build_model_from_dict(self.model_info)
        param_grid = dict(batch_size=batch_size, epochs=epochs)
        build_model = lambda: self.model
        model = KerasRegressor(build_fn=build_model, verbose=0)
        grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1, verbose=1)
        grid_result = grid.fit(self.data.x_train[0],
                    self.data.y_train[0])
        print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

    def keep_history_memory(self):
        axis_full = [*self.metrics_str, *["val_" + i for i in self.metrics_str]]
        for axis in axis_full:
            self.history_memory[axis]['history'].append(self.c_history.history[axis])
            self.history_memory[axis]['means'].append(np.mean(self.c_history.history[axis]))
            self.history_memory[axis]['min'].append(np.min(self.c_history.history[axis]))
            self.history_memory[axis]['max'].append(np.max(self.c_history.history[axis]))
     
    def plot(self, save_path=None, figsize=(16, 12), save_only=False):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if save_only:
            file_id = 0
            while os.path.isfile(save_path + str(file_id) + '_' + self.metrics_str[0] +  ".png"):
                file_id += 1
        alpha_color = [0.4]*len(self.it_order)
        alpha_color[-1] = 1.0
        for metrics in self.metrics_str:
            plt.figure(figsize=figsize)
            plt.ylabel(metrics)
            plt.xlabel('Epoch')
            for it_id in range(0, len(self.it_order)):
                plt.plot(self.history_memory[metrics]['history'][self.it_order[it_id]], color="tab:blue", alpha = alpha_color[it_id])
                plt.plot(self.history_memory["val_"+metrics]['history'][self.it_order[it_id]], color="tab:orange", alpha = alpha_color[it_id])
            plt.legend([metrics, "val_"+metrics], loc='upper left')
            x1,x2,y1,y2 = plt.axis()  
            plt.axis((x1,x2,0.0,1.0))
            if save_only:
                plt.savefig(save_path + str(file_id) + "_" + metrics + ".png")
            else:
                plt.show()
            plt.close('all')
        with open(save_path + str(file_id) + ".txt", 'w') as file:
            file.write( "R2 : " + str(round(self.history_memory['val_r2']['max'][self.it_order[-1]], 3)) + "\n")
            file.write( "RMSE : " + str(round(self.history_memory['val_rmse']['min'][self.it_order[-1]], 3)) + "\n")
            file.write( "MSE : " + str(round(self.history_memory['val_mse']['min'][self.it_order[-1]], 3)) + "\n")
    
    def save_best_model(self, save_path=None):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if self.it_order[-1]==(len(self.it_order)-1):
            self.model.save(save_path + "model")
            self.model.save_weights(save_path + "model/model_weight.h5")
    
    def combine_plot(self, DNN_list, model_name = [], save_path=None, figsize=(16, 12), save_only=False):
        color_list = ['tab:blue', 'tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:pink','tab:gray','tab:olive','tab:cyan']
        assert (len(DNN_list)+1) == len(model_name)
        if save_only:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            file_id = 0
            while os.path.isfile(save_path + str(file_id) + ".png"):
                file_id += 1
        alpha_color_d_list = []
        for DNN in DNN_list:
            assert self.metrics_str == DNN.metrics_str, "You can't plot 2 DNN with different metrics"
            alpha_color_d = [0.4]*len(DNN.it_order)
            alpha_color_d[-1] = 1.0
            alpha_color_d_list.append(alpha_color_d)
        alpha_color = [0.4]*len(self.it_order)
        alpha_color[-1] = 1.0
        for metrics in self.metrics_str:
            legend_list = []
            plt.figure(figsize=figsize)
            plt.ylabel(metrics)
            plt.xlabel('Epoch')
            for it_id in range(0, len(self.it_order)):
                plt.plot(self.history_memory[metrics]['history'][self.it_order[it_id]], color=color_list[0], alpha = alpha_color[it_id])
                plt.plot(self.history_memory["val_"+metrics]['history'][self.it_order[it_id]], color=color_list[0], alpha = alpha_color[it_id], linestyle="dashed")
            i_color = 1
            for DNN_ID in range(0, len(DNN_list)):
                DNN = DNN_list[DNN_ID]
                current_alpha = alpha_color_d_list[DNN_ID]
                for it_id in range(0, len(DNN.it_order)):
                    plt.plot(DNN.history_memory[metrics]['history'][DNN.it_order[it_id]], color=color_list[i_color], alpha = current_alpha[it_id])
                    plt.plot(DNN.history_memory["val_"+metrics]['history'][DNN.it_order[it_id]], color=color_list[i_color], alpha = current_alpha[it_id], linestyle="dashed")
                i_color=i_color+1
            legend_list.append(mpatches.Patch(color="tab:blue", label=model_name[0] + "_" + metrics, alpha=1.0))
            legend_list.append(mpatches.Patch(color="tab:blue", label=model_name[0] + "_val_" + metrics, alpha=1.0, linestyle="dashed"))
            label_id = 1
            i_color = 1
            for DNN_ID in range(0, len(DNN_list)):
                legend_list.append(mpatches.Patch(color=color_list[i_color], label=model_name[label_id] + "_" + metrics, alpha=1.0))
                legend_list.append(mpatches.Patch(color=color_list[i_color], label=model_name[label_id] + "_val_" + metrics, alpha=1.0, linestyle="dashed"))
                label_id=label_id+1
                i_color = i_color+1
            plt.legend(handles=legend_list, loc='upper left')
            x1,x2,y1,y2 = plt.axis()  
            plt.axis((x1,x2,0.0,1.0))
            if save_only:
                plt.savefig(save_path + str(file_id) + "_" + metrics + ".png")
            else:
                plt.show()
            plt.close('all')

class RF_Predictor():
    def __init__(self, data=None):
        self.data = data
        self.init_metrics()

    def init_metrics(self):
        self.metrics = [RMSE_GP, mean_absolute_error, R2_GP]
        self.metrics_str = ["rmse", "mae", "r2"]
    
    def run(self):
        self.model = RandomForestRegressor()
        self.model.fit(self.data.x_train[0], np.ravel(self.data.y_train[0].to_numpy()))
        self.pred = self.model.predict(self.data.x_test[0])
    
    def create_resdir(self, save_path, save_only):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if save_only:
            self.file_id = 0
            while os.path.isfile(save_path + str(self.file_id) +  ".txt"):
                self.file_id += 1

    def plot(self, save_path=None, save_only=True):
        self.create_resdir(save_path, save_only)
        self.accuracy = {}

        sort_index = np.argsort(self.data.y_test[0].to_numpy().flatten())
        plt.figure(num=1, figsize=(8,4), dpi=300, facecolor='w', edgecolor='k')
        plt.plot(list(range(0, len(self.data.y_test[0].to_numpy().flatten()))), self.data.y_test[0].to_numpy().flatten()[sort_index], color='y', markersize=5, label=u'Observation')
        plt.plot(list(range(0, len(self.pred))), self.pred[sort_index], 'b-', linewidth=1, label=u'Prediction')
        plt.ylim([min(self.data.y_test[0].to_numpy().flatten()) - 0.1, max(self.data.y_test[0].to_numpy().flatten())+0.1])
        plt.legend(loc='upper right', fontsize=5)
        plt.title('RMSE : ' + str(round(RMSE_GP(self.data.y_test[0].to_numpy().flatten(), self.pred), 3)) + ', R2 : ' + str(round(R2_GP(self.data.y_test[0].to_numpy().flatten(), self.pred), 3)))
        if save_only:
                plt.savefig(save_path + str(self.file_id) + ".png")
        else:
            plt.show()
        plt.close('all')

        with open(save_path + str(self.file_id) + ".txt", 'w') as file:
            file.write( "R2 : " + str(round(R2_GP(self.data.y_test[0].to_numpy().flatten(), self.pred), 3)) + "\n")
            file.write( "RMSE : " + str(round(RMSE_GP(self.data.y_test[0].to_numpy().flatten(), self.pred), 3)) + "\n")
            file.write( "MSE : " + str(round(MSE_GP(self.data.y_test[0].to_numpy().flatten(), self.pred), 3)) + "\n")
            

class DNN_auto_Classifier(DNN_Predictor):
    def __init__(self, data=None, json_models=None, metrics = ['mse', 'accuracy'], epoch=1000, iteration=5, live=True, early_stopping = [True, 300], batch_size=10, verbose=0):
        super().__init__(data=data, json_models=json_models, metrics = metrics, epoch=epoch, iteration=iteration, live=live, early_stopping = early_stopping, batch_size=batch_size, verbose=verbose)
        self.optimizer = 'adam'
        self.loss = 'mse'
        self.metrics = ['mse', Accuracy()]

    def run(self, save_path=None):
        axis_full = [*self.metrics_str, *["val_" + i for i in self.metrics_str]]
        self.history_memory = {axis :{"history" : [], "means" : [], "min" : [], "max" : []} for axis in axis_full}
        for it in range(0, self.iteration):
            self.build_model_from_dict(self.model_info)
            self.c_history = self.model.fit(self.data.x_train[0],
                        self.data.x_train[0],
                        epochs          = self.epoch,
                        batch_size      = self.batch_size,
                        verbose         = self.verbose,
                        validation_data = (self.data.x_test[0], self.data.x_test[0]),
                        callbacks = [self.callback])
            print(self.c_history.history.keys())
            self.keep_history_memory()
            self.it_order = np.argsort(self.history_memory["val_accuracy"]['max'])[::-1] # ORDER ON RMSE VALUES !
            if save_path!=None:
                self.save_best_model(save_path)
            visualkeras.layered_view(self.model, legend=True, to_file=save_path + 'model/model.png')
    
    def plot(self, save_path=None, figsize=(16, 12), save_only=False):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if save_only:
            file_id = 0
            while os.path.isfile(save_path + str(file_id) + '_' + self.metrics_str[0] +  ".png"):
                file_id += 1
        alpha_color = [0.4]*len(self.it_order)
        alpha_color[-1] = 1.0
        for metrics in self.metrics_str:
            plt.figure(figsize=figsize)
            plt.ylabel(metrics)
            plt.xlabel('Epoch')
            for it_id in range(0, len(self.it_order)):
                plt.plot(self.history_memory[metrics]['history'][self.it_order[it_id]], color="tab:blue", alpha = alpha_color[it_id])
                plt.plot(self.history_memory["val_"+metrics]['history'][self.it_order[it_id]], color="tab:orange", alpha = alpha_color[it_id])
            plt.legend([metrics, "val_"+metrics], loc='upper left')
            x1,x2,y1,y2 = plt.axis()  
            plt.axis((x1,x2,0.0,1.0))
            if save_only:
                plt.savefig(save_path + str(file_id) + "_" + metrics + ".png")
            else:
                plt.show()
            plt.close('all')


class TSNE_Visualisator():
    def __init__(self, data=None):
        self.data = data
    
    def run(self):
        if self.axes=="transpose":
            self.embedded_data = self.model.fit_transform(self.data.sub_x.transpose())
        else:
            self.embedded_data = self.model.fit_transform(self.data.sub_x)
        print(self.embedded_data.shape)
        
    
    def build_model_from_dict(self, model_dict):
        self.component = model_dict['component'] if "component" in model_dict.keys() else 3
        self.learning_rate = model_dict['learning_rate'] if "learning_rate" in model_dict.keys() else 200.0
        self.init_state = model_dict['init_state'] if "init_state" in model_dict.keys() else "random"
        self.axes = model_dict['axes'] if "axes" in model_dict.keys() else "transpose"
        self.model = TSNE(n_components=self.component, learning_rate=self.learning_rate, init=self.init_state)
    
    def plot(self, save_path, live=False, save_only=True):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if save_only:
            file_id = 0
            if self.embedded_data.shape[1] == 3:
                while os.path.isfile(save_path +  "TSNE_3D_" + str(file_id) + ".png"):
                    file_id += 1
            else:
                while os.path.isfile(save_path +  "TSNE_2D_" + str(file_id) + ".png"):
                    file_id += 1
        if self.embedded_data.shape[1] == 3:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(self.embedded_data[:,0], self.embedded_data[:,1], self.embedded_data[:,2])
            if live:
                plt.show()
            else:
                plt.savefig(save_path +  "TSNE_3D_" + str(file_id) + ".png")
        else:
            plt.plot(self.embedded_data[:,0], self.embedded_data[:,1], '.')
            if live:
                plt.show()
            else:
                plt.savefig(save_path +  "TSNE_2D_" + str(file_id) + ".png")

class SVM_Predictor():
    def __init__(self, data=None):
        self.data = data

    def build_model_from_dict(self, model_dict):
        self.tol = model_dict['tol'] if "tol" in model_dict.keys() else 1e-5
        self.random_state = model_dict['random_state'] if "random_state" in model_dict.keys() else 123
        
        self.model = make_pipeline(StandardScaler(),
                        LinearSVR(random_state=self.random_state, tol=self.tol))

    def run(self):
        self.model.fit(self.data.get_x_train(), self.data.get_y_train())
    
    def create_resdir(self, save_path, save_only):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if save_only:
            self.file_id = 0
            while os.path.isfile(save_path + str(self.file_id) +  ".txt"):
                self.file_id += 1
    
    def plot(self, save_path=None, save_only=True):
        self.create_resdir(save_path, save_only)     
        # plt.plot(np.arange(len(self.data.y_test)), self.model.predict(self.data.x_test), '-r')
        # plt.plot(np.arange(len(self.data.y_test)), self.data.y_test, 'g*')
        prediction = self.model.predict(self.data.get_x_test())
        sort_index = np.argsort(self.data.y_test[0].to_numpy().flatten())
        plt.figure(num=1, figsize=(8,4), dpi=300, facecolor='w', edgecolor='k')
        plt.plot(list(range(0, len(self.data.y_test[0].to_numpy().flatten()))), self.data.y_test[0].to_numpy().flatten()[sort_index], color='y', markersize=5, label=u'Observation')
        plt.plot(list(range(0, len(prediction))), prediction[sort_index], 'b-', linewidth=1, label=u'Prediction')
        plt.ylim([min(self.data.y_test[0].to_numpy().flatten()) - 0.1, max(self.data.y_test[0].to_numpy().flatten())+0.1])
        plt.legend(loc='upper right', fontsize=5)
        plt.title('RMSE : ' + str(round(RMSE_GP(self.data.y_test[0].to_numpy().flatten(), prediction), 3)) + ', R2 : ' + str(round(R2_GP(self.data.y_test[0].to_numpy().flatten(), prediction), 3)))
        if save_only:
                plt.savefig(save_path + str(self.file_id) + ".png")
        else:
            plt.show()
        plt.close('all')
        with open(save_path + str(self.file_id) + ".txt", 'w') as file:
            file.write( "R2 : " + str(round(R2_GP(self.data.y_test[0].to_numpy().flatten(), prediction), 3)) + "\n")
            file.write( "RMSE : " + str(round(RMSE_GP(self.data.y_test[0].to_numpy().flatten(), prediction), 3)) + "\n")
            file.write( "MSE : " + str(round(MSE_GP(self.data.y_test[0].to_numpy().flatten(), prediction), 3)) + "\n")

class UMAP_Visualisator():
    def __init__(self, data=None):
        self.data = data
        self.labels = self.data.sub_x.drop('ID', axis=1).columns.to_numpy().astype(str)
        phenotype_list = ["Li7","Na23","Mg25","P31","S34","K39","Ca43","Mn55","Fe57","Co59","Cu65","Zn66","As75","Se82","Rb85","Sr88","Mo98","Cd114"]
        self.labels_short = self.labels
        for i in range(0, len(self.labels)):
            if self.labels[i] not in phenotype_list:
                self.labels_short[i] = "SNP"
            else:
                self.labels_short[i] = "Phenotype"
        self.hover_data = pd.DataFrame({'index':np.arange(len(self.labels)),
                           'label':self.labels})
        
        
    
    def build_model_from_dict(self, model_dict):
        self.component = model_dict['component'] if "component" in model_dict.keys() else 3
        self.random_state = model_dict['random_state'] if "random_state" in model_dict.keys() else 123
        self.init_state = model_dict['init_state'] if "init_state" in model_dict.keys() else "random"
        self.axes = model_dict['axes'] if "axes" in model_dict.keys() else "transpose"
        self.model = UMAP(n_components=self.component, init=self.init_state, random_state=self.random_state)

    def run(self):
        print(self.data.sub_x.drop('ID', axis=1))
        if self.axes=="transpose":
            self.embedded_data = self.model.fit(self.data.sub_x.drop('ID', axis=1).transpose())
        else:
            self.embedded_data = self.model.fit(self.data.sub_x.drop('ID', axis=1))
        
    def plot(self, save_path, live=False, save_only=True):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if save_only:
            file_id = 0
            if self.axes=="transpose":
                while os.path.isfile(save_path + str(self.component) + "_t_" + str(file_id) + ".html"):
                    file_id += 1    
            else:
                while os.path.isfile(save_path + str(self.component) + "_" + str(file_id) +  ".html"):
                    file_id += 1        
        if self.axes=="transpose":
            umap.plot.output_file(save_path + str(self.component) + "_t_" + str(file_id) + ".html")
        else:
            umap.plot.output_file(save_path + str(self.component) + "_" + str(file_id) +  ".html")
        p = umap.plot.interactive(self.embedded_data, labels=self.labels_short, hover_data=self.hover_data, point_size=8)
        umap.plot.show(p)

class Gaussian_Process():
    def __init__(self, data=None):
        self.data = data

    def build_model_from_dict(self, model_dict = {"0" : {'type' : 'RBF', 'length' : 16384}, "1" : {'type' : 'MATERN', 'length' : 16384, 'nu' : 0.5}, 'OP' : '*'}):
        kernel = {}
        for i in range((len(model_dict)-1)):
            if model_dict[str(i)]['type'] == 'RBF':
                kernel[i] = RBF(length_scale=model_dict[str(i)]['length'], length_scale_bounds="fixed")
            if model_dict[str(i)]['type'] == 'MATERN':
                kernel[i] = Matern(nu=model_dict[str(i)]['nu'], length_scale=model_dict[str(i)]['length'], length_scale_bounds="fixed")
        if model_dict['OP'] == '+':
            self.kernel = kernel[0] + kernel[1]
        elif model_dict['OP'] == '-':
            self.kernel = kernel[0] - kernel[1]
        elif model_dict['OP'] == '*':
            self.kernel = kernel[0] * kernel[1]
        else:
            pass
        self.model = GaussianProcessRegressor(kernel=self.kernel, n_restarts_optimizer=18, normalize_y=True)
    
    def run(self, save_path=None):
        self.model.fit(self.data.x_train[0].to_numpy(), self.data.y_train[0].to_numpy())
        self.pred, self.sigma = self.model.predict(self.data.x_test[0].to_numpy(), return_std=True)
        self.pred = self.pred.flatten()

    def plot(self, save_path=None, figsize=(16, 12), save_only=False):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if save_only:
            file_id = 0
            while os.path.isfile(save_path + str(file_id) +  ".png"):
                file_id += 1
        sort_index = np.argsort(self.data.y_test[0].to_numpy().flatten())
        plt.figure(num=1, figsize=(8,4), dpi=300, facecolor='w', edgecolor='k')
        plt.plot(list(range(0, len(self.data.y_test[0].to_numpy().flatten()))), self.data.y_test[0].to_numpy().flatten()[sort_index], color='y', markersize=5, label=u'Observation')
        plt.plot(list(range(0, len(self.pred))), self.pred[sort_index], 'b-', linewidth=1, label=u'Prediction')
        plt.ylim([min(self.data.y_test[0].to_numpy().flatten()) - 0.1, max(self.data.y_test[0].to_numpy().flatten())+0.1])
        plt.legend(loc='upper right', fontsize=5)
        plt.title('RMSE : ' + str(round(RMSE_GP(self.data.y_test[0].to_numpy().flatten(), self.pred), 3)) + ', R2 : ' + str(round(R2_GP(self.data.y_test[0].to_numpy().flatten(), self.pred), 3)))
        if save_only:
                plt.savefig(save_path + str(file_id) + ".png")
        else:
            plt.show()
        plt.close('all')
        with open(save_path + str(file_id) + ".txt", 'w') as file:
            file.write( "R2 : " + str(round(R2_GP(self.data.y_test[0].to_numpy().flatten(), self.pred), 3)) + "\n")
            file.write( "RMSE : " + str(round(RMSE_GP(self.data.y_test[0].to_numpy().flatten(), self.pred), 3)) + "\n")
            file.write( "MSE : " + str(round(MSE_GP(self.data.y_test[0].to_numpy().flatten(), self.pred), 3)) + "\n")

class Gradient_boosting():
    def __init__(self, data=None):
        self.data = data

    def build_model_from_dict(self, model_dict = {"random_state" : 123}):
        self.model = GradientBoostingRegressor(random_state=model_dict['random_state'])
    
    def run(self, save_path=None):
        self.model.fit(self.data.x_train[0].to_numpy(), self.data.y_train[0].to_numpy())
        self.pred = self.model.predict(self.data.x_test[0].to_numpy())
        print(self.pred)

    def plot(self, save_path=None, figsize=(16, 12), save_only=False):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if save_only:
            file_id = 0
            while os.path.isfile(save_path + str(file_id) +  ".txt"):
                file_id += 1
        sort_index = np.argsort(self.data.y_test[0].to_numpy().flatten())
        plt.figure(num=1, figsize=(8,4), dpi=300, facecolor='w', edgecolor='k')
        plt.plot(list(range(0, len(self.data.y_test[0].to_numpy().flatten()))), self.data.y_test[0].to_numpy().flatten()[sort_index], color='y', markersize=5, label=u'Observation')
        plt.plot(list(range(0, len(self.pred))), self.pred[sort_index], 'b-', linewidth=1, label=u'Prediction')
        plt.ylim([min(self.data.y_test[0].to_numpy().flatten()) - 0.1, max(self.data.y_test[0].to_numpy().flatten())+0.1])
        plt.legend(loc='upper right', fontsize=5)
        plt.title('RMSE : ' + str(round(RMSE_GP(self.data.y_test[0].to_numpy().flatten(), self.pred), 3)) + ', R2 : ' + str(round(R2_GP(self.data.y_test[0].to_numpy().flatten(), self.pred), 3)))
        if save_only:
                plt.savefig(save_path + str(file_id) + ".png")
        else:
            plt.show()
        plt.close('all')
        with open(save_path + str(file_id) + ".txt", 'w') as file:
            file.write( "R2 : " + str(round(R2_GP(self.data.y_test[0].to_numpy().flatten(), self.pred), 3)) + "\n")
            file.write( "RMSE : " + str(round(RMSE_GP(self.data.y_test[0].to_numpy().flatten(), self.pred), 3)) + "\n")
            file.write( "MSE : " + str(round(MSE_GP(self.data.y_test[0].to_numpy().flatten(), self.pred), 3)) + "\n")

class Linear_Regression():
    def __init__(self, data=None):
        self.data = data

    def build_model_from_dict(self, model_dict = {"method" : "Linear"}):
        if model_dict["method"] == 'Linear':
            self.model = LinearRegression()
        elif model_dict["method"] == 'Lasso':
            self.model = LassoCV(cv=5)
        elif model_dict["method"] == 'Ridge':
            self.model = RidgeCV(cv=5)
        elif model_dict["method"] == 'ElasticNet':
            self.model = ElasticNetCV(cv=5)
        else:
            raise NotImplementedError()
        self.method = model_dict["method"]
    
    def run(self, save_path=None):
        self.model.fit(self.data.x_train[0].to_numpy(), self.data.y_train[0].to_numpy())
        self.pred = self.model.predict(self.data.x_test[0].to_numpy())
        self.pred = self.pred.flatten()

    def plot(self, save_path=None, figsize=(16, 12), save_only=False):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if save_only:
            file_id = 0
            while os.path.isfile(save_path + self.method + "_" + str(file_id) + ".png"):
                file_id += 1
        sort_index = np.argsort(self.data.y_test[0].to_numpy().flatten())
        plt.figure(num=1, figsize=(8,4), dpi=300, facecolor='w', edgecolor='k')
        plt.plot(list(range(0, len(self.data.y_test[0].to_numpy().flatten()))), self.data.y_test[0].to_numpy().flatten()[sort_index], color='y', markersize=5, label=u'Observation')
        plt.plot(list(range(0, len(self.pred))), self.pred[sort_index], 'b-', linewidth=1, label=u'Prediction')
        plt.ylim([min(self.data.y_test[0].to_numpy().flatten()) - 0.1, max(self.data.y_test[0].to_numpy().flatten())+0.1])
        plt.legend(loc='upper right', fontsize=5)
        plt.title('RMSE : ' + str(round(RMSE_GP(self.data.y_test[0].to_numpy().flatten(), self.pred), 3)) + ', R2 : ' + str(round(R2_GP(self.data.y_test[0].to_numpy().flatten(), self.pred), 3)))
        if save_only:
                plt.savefig(save_path + self.method + "_" + str(file_id) + ".png")
        else:
            plt.show()
        with open(save_path + str(file_id) + ".txt", 'w') as file:
            file.write( "R2 : " + str(round(R2_GP(self.data.y_test[0].to_numpy().flatten(), self.pred), 3)) + "\n")
            file.write( "RMSE : " + str(round(RMSE_GP(self.data.y_test[0].to_numpy().flatten(), self.pred), 3)) + "\n")
            file.write( "MSE : " + str(round(MSE_GP(self.data.y_test[0].to_numpy().flatten(), self.pred), 3)) + "\n")
        plt.close('all')
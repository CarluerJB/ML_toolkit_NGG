import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from library_python_script.tools.utility import generate_ID_list_inter, K_random_index_generator
import matplotlib.pyplot as plt
import os

class FileManager():
    # CONSTRUCTORS
    # Constructor for x/y filemanager
    def __init__(self, x_data_path:str, y_data_path:str, ID_data_path:str): 
        self.x_data_path = x_data_path
        self.y_data_path = y_data_path
        self.ID_data_path = ID_data_path
        self.x_header=None
        self.y_header=0
        self.x_separator=" "
        self.y_separator=";"
        self.class_type = "full"
        self.sub_x = None
        self.sub_y = None
    
    # FUNCTIONS
    def setYParameters(self, y_filemanager):
        self.y_data_path = y_filemanager.y_data_path
        self.y_header=y_filemanager.y_header
        self.y_separator=y_filemanager.y_separator
        self.class_type = "full"
    
    def setXParameters(self, x_filemanager):
        self.x_data_path = x_filemanager.x_data_path
        self.ID_data_path = x_filemanager.ID_data_path
        self.x_header = x_filemanager.header
        self.x_separator = x_filemanager.x_separator
        self.class_type = x_filemanager.class_type
    
    def loadX(self, x_filemanager=None, header=None, sep = " "):
        self.x_header = header
        self.x_separator = sep
        if x_filemanager!=None:
            self.setXParameters(x_filemanager)
        assert self.class_type!="y", "You need to set X parameters to be able to load data !"
        if self.x_data_path.split('.')[1]=="npy":
            self.x_full = np.load(self.x_data_path)
            self.x_full = pd.DataFrame(self.x_full)
        else:
            self.x_full = pd.read_table(self.x_data_path, header=self.x_header, sep = self.x_separator)
        self.nb_SNP_full = len(self.x_full.columns)
        self.nb_individuals_full = len(self.x_full.index)
    
    def loadY(self, y_filemanager=None, header=0, sep = ";"):
        self.y_header = header
        self.y_separator = sep
        if y_filemanager!=None:
            self.setYParameters(y_filemanager)
        assert self.class_type!="x", "You need to set Y parameters to be able to load data !"
        self.y_full = pd.read_csv(self.y_data_path, header = self.y_header, sep = self.y_separator)
    
    def loadID(self, ID_filemanager=None, dtype=np.integer):
        self.ID_full = np.loadtxt(self.ID_data_path, dtype=dtype)

    def loadData(self):
        if self.class_type=="full":
            self.loadX()
            self.loadY()
            self.loadID()
        elif(self.class_type=="x"):
            self.loadX()
        elif(self.class_type=="y"):
            self.loadY()
        else:
            raise NotImplementedError
    
    def buildSubXData(self, method:str, sub_id=None, file="", append=False, axis=1, normalize=True, random=False, reduce_dim=None):
        # For create subtable in 1D
        if method == "std":
            self.sub_x = self.x_full.loc[sub_id['line'][0] : sub_id['line'][1], [*sub_id['col'], 'ID']] if 'line' in sub_id.keys() else self.x_full.loc[:, [*sub_id['col'], 'ID']]
            if normalize==True:
                for col in sub_id['col']:
                    self.sub_x[col] = (self.sub_x[col] - self.sub_x[col].mean()) / self.sub_x[col].std()
        elif method == "std-rem":
            sub_x_temp = self.y_full.loc[sub_id['line'][0] : sub_id['line'][1], list(self.y_full.columns.drop(sub_id['col']))]
            if self.sub_x==None:
                self.sub_x = pd.DataFrame({'ID' : self.ID_full})
            else:
                self.sub_x['ID'] = self.ID_full
            self.sub_x = self.sub_x.merge(sub_x_temp, on='ID', how='inner')
        elif method=="by_top":
            if sub_id==None or (np.isscalar(sub_id)):
                KId = sub_id
                if random==True:
                    sub_id = K_random_index_generator(nb_SNP=self.nb_SNP_full, nb_ktop=20000, inter=False)['K'].to_numpy()
                else:
                    sub_id=np.asarray(np.loadtxt(file, dtype=int))
                if np.isscalar(KId):
                    sub_id=sub_id[:KId]
            if append:
                self.sub_x = pd.concat([self.sub_x, self.x_full.iloc[:, sub_id].reset_index(drop=True)], axis=axis)
            else:
                self.sub_x = self.x_full.iloc[:, sub_id]
        # For create subtable in 2D
        elif method=="product":
            if (sub_id==None) or (np.isscalar(sub_id)):
                KId = sub_id
                if random==True:
                    sub_id = K_random_index_generator(nb_SNP=self.nb_SNP_full, nb_ktop=20000)
                else:
                    sub_id=pd.read_table(file, header=None, names=['K'], dtype={'K' : int})
                if np.isscalar(KId):
                    sub_id = sub_id.head(KId)
            ID_list = generate_ID_list_inter(sub_id, len(self.x_full.columns)) # TODO import generate SNP list inter
            ID1 = self.x_full[ID_list['i']]
            ID2 = self.x_full[ID_list['j']]
            ID1 = ID1.T.reset_index(drop=True).T
            ID2 = ID2.T.reset_index(drop=True).T
            if append:
                self.sub_x = pd.concat([self.sub_x, ID1.mul(ID2)], axis=axis)
            else:
                self.sub_x = ID1.mul(ID2)
        else:
            raise NotImplementedError
        if normalize==True:
            self.sub_x = (self.sub_x - self.sub_x.mean()) / self.sub_x.std()
        if reduce_dim!=None:
            X = self.sub_x.to_numpy()
            XXt = np.matmul(X, X.transpose())
            eigenvalues, eigenvectors = np.linalg.eig(XXt)
            eigenvalues = eigenvalues.real
            eigenvectors = eigenvectors.real
            idx_sort = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[idx_sort]
            eigenvectors = eigenvectors[:, idx_sort]
            newX = np.matmul(X, np.matmul(X.transpose(), eigenvectors))
            if reduce_dim < newX.shape[1]:
                newX = newX[:, :reduce_dim]
            self.sub_x = pd.DataFrame(newX)
    
    def buildSubYData(self, method:str, sub_id=[0, 1152],append=False, axis=0, normalize=True, categorize=None, nb_cluster=2, show_cluster=False, save_path=None):
        if method == "std":
            self.sub_y = self.y_full.loc[sub_id['line'][0] : sub_id['line'][1], [*sub_id['col'], 'ID']] if 'line' in sub_id.keys() else self.y_full.loc[:, [*sub_id['col'], 'ID']]
            self.y_name = sub_id['col'][0]
            if normalize==True:
                for col in sub_id['col']:
                    self.sub_y[col] = (self.sub_y[col] - self.sub_y[col].mean()) / self.sub_y[col].std()
            if categorize!=None: # Create categorie from qauntile values
                if categorize == "by_quantile":
                    sub_y_copy = self.sub_y.copy(deep=True)
                    for col in sub_id['col']:
                        if nb_cluster==5:
                            quantile_list = self.sub_y[col].quantile([0.20,0.40,0.60,0.80,1.0])
                            sub_y_copy.loc[(self.sub_y[col] <= quantile_list[1.0]) & (self.sub_y[col] >= quantile_list[0.80]), col] = 4
                            sub_y_copy.loc[(self.sub_y[col] <= quantile_list[0.80]) & (self.sub_y[col] >= quantile_list[0.60]), col] = 3
                            sub_y_copy.loc[(self.sub_y[col] <= quantile_list[0.60]) & (self.sub_y[col] >= quantile_list[0.40]), col] = 2
                            sub_y_copy.loc[(self.sub_y[col] <= quantile_list[0.40]) & (self.sub_y[col] >= quantile_list[0.20]), col] = 1
                            sub_y_copy.loc[self.sub_y[col] <= quantile_list[0.20], col] = 0
                            self.nb_class=[0,1,2,3,4]
                        elif nb_cluster==3:
                            quantile_list = self.sub_y[col].quantile([0.33,0.66,1.0])
                            sub_y_copy.loc[(self.sub_y[col] <= quantile_list[1.0]) & (self.sub_y[col] >= quantile_list[0.66]), col] = 2
                            sub_y_copy.loc[(self.sub_y[col] <= quantile_list[0.66]) & (self.sub_y[col] >= quantile_list[0.33]), col] = 1
                            sub_y_copy.loc[self.sub_y[col] <= quantile_list[0.33], col] = 0
                            self.nb_class=[0,1,2]
                        self.sub_y = sub_y_copy.copy(deep=True)
                elif categorize == "by_kmeans":
                    for col in sub_id['col']:
                        order = np.argsort(self.sub_y[col].to_numpy())
                        values = np.sort(self.sub_y[col].to_numpy())
                        kmeans = KMeans(n_clusters=nb_cluster, random_state=0).fit(self.sub_y[col].to_numpy().reshape(-1, 1))
                        self.sub_y.loc[:, col] = kmeans.labels_
                        self.nb_class=list(range(kmeans.labels_.max()+1))
                        u_labels = np.unique(kmeans.labels_)
                        print(u_labels)
                        if(show_cluster):
                            if not os.path.exists(save_path):
                                os.makedirs(save_path)
                            plt.scatter(range(len(self.sub_y[col].to_numpy())), values, marker='x')
                            plt.scatter(range(len(self.sub_y[col].to_numpy())), values , marker = 'o', c = self.sub_y[col].to_numpy()[order])
                            plt.savefig(save_path + "distrib_clust_" + categorize + "_" + str(nb_cluster) + ".png")
                else:
                    print("[ERROR] Method \"" + categorize + "\" is unknown or is not compatible with current parameter")
                    exit(0)
        elif method == "identity":
            self.sub_y = self.y_full
        elif method == "min_max_l_delim":
            if append:
                self.sub_y = pd.concat([self.sub_y, self.y_full[sub_id[0] : sub_id[1]]], axis=axis)
            else:
                self.sub_y = self.y_full[sub_id[0] : sub_id[1]]
        else:
            raise NotImplementedError
        
        

    def splitTrainVal(self, train_ID=None, ratio=0.8, random_state=123, categorize_done=None, normalize=True, categorize=None, nb_cluster=3):
        self.sub_x['ID'] = self.ID_full
        data = self.sub_x.merge(self.sub_y, on='ID', how='inner')
        if train_ID==None:
            data = data.drop('ID', axis = 1)
            data_train = data.sample(frac = ratio, axis = 0, random_state = random_state)
            data_test = data.drop(data_train.index)
        else:
            data_train = data[data['ID'].isin(train_ID)]
            data = data.drop('ID', axis = 1)
            data_test = data.drop(data_train.index)
        if normalize:
            mean = data_train.mean()
            std  = data_train.std()
            if categorize_done!=None:
                mean[self.y_name] = 0.0
                std[self.y_name] = 1.0
            data_train = (data_train-mean)/std
            data_test = (data_test-mean)/std
        if categorize == "by_quantile":
            data_train_copy = data_train.copy(deep=True)
            data_test_copy = data_test.copy(deep=True)
            if nb_cluster==5:
                quantile_list = data_train[self.y_name].quantile([0.20,0.40,0.60,0.80,1.0])
                data_train_copy.loc[(data_train[self.y_name] <= quantile_list[1.0]) & (data_train[self.y_name] >= quantile_list[0.80]), self.y_name] = 4
                data_train_copy.loc[(data_train[self.y_name] <= quantile_list[0.80]) & (data_train[self.y_name] >= quantile_list[0.60]), self.y_name] = 3
                data_train_copy.loc[(data_train[self.y_name] <= quantile_list[0.60]) & (data_train[self.y_name] >= quantile_list[0.40]), self.y_name] = 2
                data_train_copy.loc[(data_train[self.y_name] <= quantile_list[0.40]) & (data_train[self.y_name] >= quantile_list[0.20]), self.y_name] = 1
                data_train_copy.loc[data_train[self.y_name] <= quantile_list[0.20], self.y_name] = 0
                quantile_list = data_test[self.y_name].quantile([0.20,0.40,0.60,0.80,1.0])
                data_test_copy.loc[(data_test[self.y_name] <= quantile_list[1.0]) & (data_test[self.y_name] >= quantile_list[0.80]), self.y_name] = 4
                data_test_copy.loc[(data_test[self.y_name] <= quantile_list[0.80]) & (data_test[self.y_name] >= quantile_list[0.60]), self.y_name] = 3
                data_test_copy.loc[(data_test[self.y_name] <= quantile_list[0.60]) & (data_test[self.y_name] >= quantile_list[0.40]), self.y_name] = 2
                data_test_copy.loc[(data_test[self.y_name] <= quantile_list[0.40]) & (data_test[self.y_name] >= quantile_list[0.20]), self.y_name] = 1
                data_test_copy.loc[data_test[self.y_name] <= quantile_list[0.20], self.y_name] = 0
                self.nb_class=[0,1,2,3,4]
            elif nb_cluster==3:
                quantile_list = data_train[self.y_name].quantile([0.33,0.66,1.0])
                data_train_copy.loc[(data_train[self.y_name] <= quantile_list[1.0]) & (data_train[self.y_name] >= quantile_list[0.66]), self.y_name] = 2
                data_train_copy.loc[(data_train[self.y_name] <= quantile_list[0.66]) & (data_train[self.y_name] >= quantile_list[0.33]), self.y_name] = 1
                data_train_copy.loc[data_train[self.y_name] <= quantile_list[0.33], self.y_name] = 0
                quantile_list = data_test[self.y_name].quantile([0.33,0.66,1.0])
                data_test_copy.loc[(data_test[self.y_name] <= quantile_list[1.0]) & (data_test[self.y_name] >= quantile_list[0.66]), self.y_name] = 2
                data_test_copy.loc[(data_test[self.y_name] <= quantile_list[0.66]) & (data_test[self.y_name] >= quantile_list[0.33]), self.y_name] = 1
                data_test_copy.loc[data_test[self.y_name] <= quantile_list[0.33], self.y_name] = 0
                self.nb_class=[0,1,2]
            data_train = data_train_copy.copy(deep=True)
            data_test = data_test_copy.copy(deep=True)
        elif categorize == "by_kmeans":
            kmeans = KMeans(n_clusters=nb_cluster, random_state=0).fit(data_train[self.y_name].to_numpy().reshape(-1, 1))
            data_train.loc[:, self.y_name] = np.asarray(kmeans.labels_)
            self.nb_class=list(range(kmeans.labels_.max()+1))
            u_labels = np.unique(kmeans.labels_)
            print(u_labels)
            kmeans_2 = KMeans(n_clusters=nb_cluster, random_state=0).fit(data_test[self.y_name].to_numpy().reshape(-1, 1))
            data_test.loc[:, self.y_name] = np.asarray(kmeans_2.labels_)
            self.nb_class=list(range(kmeans_2.labels_.max()+1))
            u_labels = np.unique(kmeans_2.labels_)
            print(u_labels)

        self.x_train = [data_train.drop(list(self.sub_y.columns.drop('ID').values), axis = 1)]
        if (categorize_done!=None) or categorize!=None:
            self.y_train = [data_train[list(self.sub_y.columns.drop('ID').values)].astype(np.int32())]
        else:
            self.y_train = [data_train[list(self.sub_y.columns.drop('ID').values)]]
        self.x_test = [data_test.drop(list(self.sub_y.columns.drop('ID').values), axis = 1)]
        if (categorize_done!=None) or categorize!=None:
            self.y_test = [data_test[list(self.sub_y.columns.drop('ID').values)].astype(np.int32())]
        else:
            self.y_test = [data_test[list(self.sub_y.columns.drop('ID').values)]]
    
    def encodeYonehotEncoding(self):
        df = pd.DataFrame({elem : [] for elem in self.nb_class})
        for elem in self.get_y_train():
            val = {elem2 : 0 for elem2 in self.nb_class}
            val[elem] = 1
            df=df.append(val, ignore_index=True)
        self.y_train_onehot = [df.copy(deep=True)]

        df = pd.DataFrame({elem : [] for elem in self.nb_class})
        for elem in self.get_y_test():
            val = {elem2 : 0 for elem2 in self.nb_class}
            val[elem] = 1
            df=df.append(val, ignore_index=True)
        self.y_test_onehot = [df.copy(deep=True)]

    def get_x_train(self):
        return self.x_train[0].values
    
    def get_x_test(self):
        return self.x_test[0].values
    
    def get_y_train(self, ravel=True, onehot=False):
        if ravel:
            if onehot:
                return self.y_train_onehot[0].values.ravel()
            else:
                return self.y_train[0].values.ravel()
        else:
            if onehot:
                return self.y_train_onehot[0].values
            else:
                return self.y_train[0].values
    
    def get_y_test(self, ravel=True, onehot=False):
        if ravel:
            if onehot:
                return self.y_test_onehot[0].values.ravel()
            else:
                return self.y_test[0].values.ravel()
        else:
            if onehot:
                return self.y_test_onehot[0].values
            else:
                return self.y_test[0].values
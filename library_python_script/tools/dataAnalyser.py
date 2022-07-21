import hdbscan
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

class DataAnalyser():
    def __init__(self, savepath=None):
        self.savepath=savepath
        

    def Hdbscan(self, data):
        if not os.path.exists(self.savepath):
            os.makedirs(self.savepath)
        clusterer = hdbscan.HDBSCAN()
        cluster_labels = clusterer.fit(data.reshape(-1, 1))
        with open(self.savepath + "hdbscan_cluster_nb.txt") as file:
            file.write("Nb cluster : " + str(cluster_labels.labels_.max()))
    
    def DensityPlot(self, data):
        if not os.path.exists(self.savepath):
            os.makedirs(self.savepath)
        sns.set_style('whitegrid')
        sns.kdeplot(np.array(data), bw=0.5)
        plt.savefig(self.savepath + "density_plot" + ".png")

    

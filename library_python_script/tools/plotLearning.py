

from tensorflow import keras
from IPython.display import clear_output
import matplotlib.pyplot as plt




class PlotLearning(keras.callbacks.Callback):
    """
    Callback to plot the learning curves of the model during training.
    """
    def on_train_begin(self, logs={}):
        self.metrics = {}
        for metric in logs:
            self.metrics[metric] = []


    def on_epoch_end(self, epoch, logs={}):
        # Storing metrics
        for metric in logs:
                if metric in self.metrics:
                    self.metrics[metric].append(logs.get(metric))
                else:
                    self.metrics[metric] = [logs.get(metric)]
        if epoch%100 == 0:
            # Plotting
            metrics = [x for x in logs if 'val' not in x]
            if(epoch==0):
                self.figure, self.axs = plt.subplots(1, len(metrics), figsize=(15,5))
            clear_output(wait=False)

            for i, metric in enumerate(metrics):
                self.axs[i].plot(range(1, epoch + 2),
                            self.metrics[metric],
                            label=metric,
                            color='red')
                if logs['val_' + metric]:
                    self.axs[i].plot(range(1, epoch + 2),
                                self.metrics['val_' + metric],
                                label='val_' + metric,
                                color='blue')

                if(epoch==0):
                    self.axs[i].set_ylim([0.0, 1.0])
                    self.axs[i].legend()
                    self.axs[i].grid()

            plt.tight_layout()
            plt.show(block=False)
            plt.pause(0.0001)

    def on_train_end(self, logs={}):
        plt.close('all')

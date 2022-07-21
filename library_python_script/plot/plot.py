import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def heatplot(data, x_header, y_header, title="heatplot"):
    fig, ax = plt.subplots()
    im = ax.imshow(data)
    ax.set_xticks(np.arange(len(x_header)))
    ax.set_yticks(np.arange(len(y_header)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(x_header)
    ax.set_yticklabels(y_header)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(y_header)):
        for j in range(len(x_header)):
            text = ax.text(j, i, data[i, j],
                           ha="center", va="center", color="w")

    ax.set_title(title)
    fig.tight_layout()
    plt.show()

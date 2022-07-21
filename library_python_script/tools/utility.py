import numpy as np
import pandas as pd
import random

def generate_ID_list_inter(ID_list, size):
    ID_list['i'] = int(size) -2 - np.floor(np.sqrt(-8*ID_list['K'] + 4 * int(size) * (int(size)-1) - 7) / 2 - 0.5)
    ID_list['i'] = ID_list['i'].astype('int')
    ID_list['j'] = ID_list['K'] + ID_list['i'] + 1 - int(size) * (int(size)-1) / 2 + (int(size) - ID_list['i']) * ((int(size) - ID_list['i']) - 1) / 2
    ID_list['j'] = ID_list['j'].astype('int')
    return ID_list

def K_random_index_generator(nb_SNP=0, nb_ktop=0, inter=True):
    if inter:
        nb_interaction = int(nb_SNP*(nb_SNP-1)/2)
        K_id = random.sample(range(0, nb_interaction), nb_ktop)
    else:
        K_id = random.sample(range(0, nb_SNP), nb_ktop)
    K_pd = pd.DataFrame(K_id, columns = ['K'], dtype=int)
    return K_pd

def load_parameters(parameters):
    # TODO !
    pass

def OneHotEncoding(data, nb_class):
    labels = nb_class
    new_array = np.array([[0]*len(labels)]*len(data))
    for elem in range(len(data)):
        new_array[elem, data[elem]] = 1
    return new_array
        
import pandas as pd
import numpy as np
import csv
from hyperopt import hp

class helper_functions():
    def get_data():
        #Load Training Data
        owli_data_train = pd.read_pickle('/axp/rim/csgsnml/dev/ajkent/DATA/real_time_train_full.pkl')
        X_train = owli_data_train[[x for x in owli_data_train.columns if x not in ['next_action','next_index']]].copy()
        Y_train = pd.get_dummies(owli_data_train.next_action) #converts the labels to one-hot-encoded
        #Load Testing Data
        owli_data_test = pd.read_pickle('/axp/rim/csgsnml/dev/ajkent/DATA/real_time_test.pkl')
        labels_new = pd.read_pickle('/axp/rim/csgsnml/dev/ajkent/DATA/labels_drop.pkl')
        labels_new['next_action_list'] = labels_new['next_action_list'].str.split(';').apply(lambda x: list(filter(None,[val.strip() if val.strip() != '' else None for val in x])))
        merged = pd.merge(owli_data_test, labels_new[['next_index','next_action_list']], on='next_index')
        merged['next_action_list'] = merged['next_action_list'].str.join(';')
        owli_test = merged.drop_duplicates(subset = ['next_index', 'next_action_list'])
        X_test = owli_test[[x for x in owli_test.columns if x not in ['actual_action_list','next_index','next_action','next_action_list']]].copy()
        Y_test = pd.get_dummies(owli_test.next_action)
        nb_classes = Y_train.shape[1]
        input_shape=(X_train.shape[1],)
        return (nb_classes, input_shape, X_train, X_test, Y_train, Y_test,owli_test,labels_new)

    def define_search_space():
        space = {
            'nb_layers': hp.quniform('nb_layers', 1,7, 1), #Layers from 1-7, incrementing by 1
            'nb_neurons': hp.quniform('nb_neurons', 64, 1024, 32), #NB neurons from 64-1024, incrementing by 64
            'activation': hp.choice('activation', ['relu', 'elu', 'tanh', 'sigmoid']), #Chose one of the 4 activations functions for the layers
            'optimizer': hp.choice('optimizer', ['rmsprop', 'adam', 'sgd', 'adagrad',
                      'adadelta', 'adamax', 'nadam']) #Chose one of the 7 optimizers
                }
	
        return space
    def create_ouput_file():
        out_file = './neural_network.csv'
        of_connection = open(out_file, 'w')
        writer = csv.writer(of_connection)
        writer.writerow(['training_accuracy','top_1_val_accuracy','top_3_val_accuracy', 'params', 'iteration', 'train_time'])
        of_connection.close()
        return out_file
    def write_to_output_file(file_name = None, results = None):
        of_connection = open(file_name, 'a')
        writer = csv.writer(of_connection)
        writer.writerow(results)

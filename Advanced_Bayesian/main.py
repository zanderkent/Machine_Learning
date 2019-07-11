from hyperopt import fmin,tpe,Trials,fmin,hp,STATUS_OK
import pandas as pd
import numpy as np
from network import Network
from custom_eval_functions import custom_accuracy
from timeit import default_timer as timer
from helper_functions import helper_functions

gpus = 2
MAX_EVALS = 150
N_FOLDS = 5
#load data
nb_classes, input_shape, X_train, X_test, Y_train, Y_test,entire_test,labels_new = helper_functions.get_data()

out_file = helper_functions.create_ouput_file()

def objective(params, n_folds = N_FOLDS):
    """Objective function for MLP Hyperparameter Optimization"""
    # Keep track of evals
    global ITERATION 
    ITERATION += 1
    start = timer()
    #When using Hyperpot to generate our lists, it creates them as floats, to we are converting them to
    #integers so that they will work on our Neural Network
    for parameter_name in ['nb_neurons', 'nb_layers']:
        params[parameter_name] = int(params[parameter_name])
    #Create and compile our network, from the parameters that the TPE chooses. 
    model = Network(params, input_shape=input_shape, nb_classes = nb_classes,output_activation='softmax',
                     loss_function='categorical_crossentropy', gpus=gpus, patience=5)
    model.compile_model()
    model.cross_validate_model(X_train = X_train,Y_train=Y_train,num_folds=n_folds,
                               epochs=1000,training_verbose=0, testing_verbose=0,
                               return_val=1, onehot_encode=True)
    cross_val_score = model.accuracy
    #Here is the custom accuracy function. If you don't need to use this
    #you can just use the cross_val_score instead
    preds = custom_accuracy.GetPrediction(model.model,X_test,entire_test,Y_test)
    new_preds = preds.merge(labels_new[['next_index', 'next_action_list']], on = ['next_index'])
    new_preds['actual_action_list'] = new_preds['next_action_list']
    top_3_acc, top_1_acc = custom_accuracy.suppresed_top_accuracy(new_preds)
    #This allows us to keep track of how long it takes to run the 5 fold CV on our NN.
    #This is important if we need to make comprimises on speed VS accuracy.
    #The more nodes/layers that you have, the longer it takes for the NN to train/run
    run_time = timer() - start
    
    # We take 1 - our accuracy to get how far away our model is from 100% accuracy. 
    loss = 1 - top_1_acc.mean()
    results = [cross_val_score,top_1_acc.mean(),top_3_acc.mean(), params, ITERATION, run_time]
    helper_functions.write_to_output_file(file_name = out_file, results = results)
    # Dictionary with information for evaluation
    return {'loss': loss, 'params': params, 'iteration': ITERATION,
            'train_time': run_time, 'status': STATUS_OK}

space = helper_functions.define_search_space()
#from hyperopt.pyll.stochastic import sample
#print(sample(space))
# optimization algorithm
tpe_algorithm = tpe.suggest
# Keep track of results
bayes_trials = Trials()
# Global variable
global  ITERATION
ITERATION = 0
best = fmin(fn = objective, space = space, algo = tpe_algorithm, 
            max_evals = MAX_EVALS, trials = bayes_trials,show_progressbar=False, 
			rstate = np.random.RandomState(50))
print(best)
#if __name__ == '__main__':
#    main()

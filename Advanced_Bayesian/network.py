from keras.models import Sequential #used for the FFNN
from keras.layers import Dense, Dropout #used for the FFNN
from keras.callbacks import EarlyStopping #used for the FFNN
from keras.utils import multi_gpu_model #used for multi_gpu enablement
import tensorflow as tf #Used to do multi_gpu
import traceback #Try and except print issues found
from keras.utils import to_categorical
import sys #Flush if needed
import numpy as np
from sklearn.model_selection import StratifiedKFold

class Network():
    """Represent a network and let us operate on it.
    Currently only works for an MLP.
    """

    def __init__(self, network=None, nb_classes=None, input_shape=None,output_activation='softmax',
                 loss_function='categorical_crossentropy', gpus=2, patience=5):
        """Initialize our network.
        Args:
            network (dict): Parameters for the network, includes:
                nb_neurons (int): any number between 1 and inf (needs to be able to fit into memory)
                nb_layers (int): any number between 1 and inf (needs to be able to fit into memory)
                activation (string): one in this list ['relu', 'elu', 'tanh', 'sigmoid']
                optimizer  (string): one in this list ['rmsprop', 'adam', 'sgd', 'adagrad','adadelta', 'adamax', 'nadam']
            nb_classes (int) = output vector for network, 1 if binary n if categorical
            input_shape (tuple): (num_features,)
            output_activation (string): default = softmax for categorical, use sigmoid for multilabel or binary classification
            loss_function: default = categorical_crossentropy options (binary_crossentropy)
            gpus (int): if you are running on a gpu, change this number to the number of gpus
            patience (int): number of epochs to monitor before stopping early
        """
        self.network = network # (dic): represents MLP network parameters
        self.nb_classes = nb_classes
        self.input_shape = input_shape
        self.output_activation = output_activation
        self.loss_function = loss_function
        self.num_gpus = gpus
        self.early_stopper = EarlyStopping(patience=patience)
        self.model = False
        self.accuracy = 0
    def compile_model(self):
        """Compiles our network using the parameters passed in the init function
            This returns a keras compiled model, so you can perform any functions on this that you can on any keras model
            like  model.fit, model.summary model.evaluate etc. if you want to use it in the cross valiate function, don't run this seperatly."""
        nb_layers = self.network['nb_layers']
        nb_neurons = self.network['nb_neurons']
        activation = self.network['activation']
        optimizer = self.network['optimizer']
        num_gpus = self.num_gpus #you can either hard code this number or use all of the gpus

        model = Sequential()
        # Add each layer.
        for neurons in range(nb_layers):
            if neurons == 0: #This is the input layer
                model.add(Dense(nb_neurons, activation=activation, input_shape=self.input_shape,kernel_initializer ='glorot_normal', name='input_layer'))
                model.add(Dropout(0.2, name='dropout_input_layer'.format(neurons)))
            else: #All other layers
                model.add(Dense(nb_neurons, activation=activation, kernel_initializer='glorot_normal', name='layer_{}'.format(neurons)))
                model.add(Dropout(0.2, name='droput_layer_{}'.format(neurons)))  # hard-coded dropout
        model.add(Dense(self.nb_classes, activation=self.output_activation, name='output_layer'))
        #If there is available GPUs on the system it will use them.
        try:
            if num_gpus<=1:
                model.compile(loss=self.loss_function,  optimizer=optimizer,metrics=['categorical_accuracy'])
            else:
                with tf.device("/cpu:0"):
                    model = multi_gpu_model(model, gpus=num_gpus)
                    model.compile(loss=self.loss_function, optimizer=optimizer,metrics=['categorical_accuracy'])
        except Exception:
            print(traceback.print_exc())
            model.compile(loss='binary_crossentropy', optimizer=optimizer,metrics=['categorical_accuracy'])
            print("Using CPU ONLY to train")
            sys.stdout.flush()
        self.model = model
    def cross_validate_model(self,X_train=None, Y_train=None,num_folds=5,
                              epochs=100,training_verbose=0, testing_verbose=1,
                             return_val=1, onehot_encode=True):
        """ Cross validate the model using Kfolds, returns average accuracy or loss
            args
                model: model created from compile method
                X_train (vector) is the features you want to train on, this should match your shape you passed to create the object
                Y_train (vector) target vector you want to train towards
                onehot_encode (binary):  Put to false if your data is NOT one hot encoded and needs to be, put True for binary. 
                early_stopper (object): passed from init
                num_folds(int): number of times to cross validate
                epochs (int): how many interations each model should train towards
                verbose (int): default is zero, if you want the progress of each NN printed out set this to 1
                return_val (int): default is 1 which is accuracy, 0 is loss, 50 returns the whole list"""
        if not self.model:
            self.compile_model()
        model = self.model
        early_stopper = self.early_stopper
        Kfold = StratifiedKFold(n_splits=num_folds)
        if not type(X_train) == np.ndarray:
            X_train = np.array(X_train)
        if not type(Y_train) == np.ndarray:
            Y_train = np.array(Y_train)
        if onehot_encode:
            generator = Kfold.split(X_train, Y_train[:,0])
        else:
            generator = Kfold.split(X_train, Y_train)
            Y_train = to_categorical(Y_train, len(set(Y_train)))
        cross_validated_scores = []
        for val in generator:
            model.fit(x=X_train[val[0]], y=Y_train[val[0]], 
                      batch_size=512, 
                      validation_data = [X_train[val[1]],Y_train[val[1]]],
                     epochs=epochs, verbose=training_verbose, callbacks=[early_stopper])
            cross_validated_scores.append(model.evaluate(X_train[val[1]],Y_train[val[1]],verbose=testing_verbose))
        if return_val == 50:
            self.accuracy = np.array(cross_validated_scores)
        else:
            self.accuracy = np.array(cross_validated_scores)[:,return_val].mean()

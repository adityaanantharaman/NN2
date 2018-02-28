# NN2
neural network with 2 layers(1 hidden layer)
this is a program written in octave that can be used to train a model on the given dataset of images.

pass the train data, target values and the number of classes present to the function [train].
the model will be trained and progress will be showed. the parameters after training will be returned by the [train] function and must be saved in [theta1,theta2].
now pass the test data and the [theta1,theta2] to predict and it returns the predictions of all the test data using the trained model.
the number of nurons in the hidden layer can be changed by the user in the [train] function.
it is best to set the number of neurons in the hidden layer as the avg of the neurons in the input and output layer.

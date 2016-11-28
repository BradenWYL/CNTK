# Import the relevant components
import numpy as np
import sys
import os
from cntk import Trainer, cntk_device, StreamConfiguration, learning_rate_schedule, UnitType
from cntk.device import cpu, set_default_device
from cntk.learner import sgd
from cntk.ops import *
from cntk.utils import get_train_eval_criterion, get_train_loss

import parseToTree as ptt
import FeedForwardNet as ffn

printLog = True

# main program
# read in training and testing files
parsed_train = input ("What is the parsed file for training? ")
highlighted_train = input ("What is the highlighted file for training? ")
parsed_test = input ("What is the parsed file testing? ")
highlighted_test = input ("What is the highlighted file for testing? ")

# extract features and construct input and output matrices
print ("------")
input_train = np.array([])
output_train = np.array([])
input_test = np.array([])
output_test = np.array([])

# handle training data
datas = ptt.handleFile (parsed_train, highlighted_train)
for data in datas:
	temp_input = np.array([])
	temp_label = np.zeros((1, 2))
	for i in data:
		if (data [i].vector().shape[0]) == 1:
			if temp_input.size == 0:
				temp_input = data [i].vector()
				temp_label[0,:] = [0, data[i].label]
				print(temp_label)
				print(str(temp_label.shape))
			else:
				temp_input = np.concatenate((temp_input, data [i].vector()), axis=0)
				temp_label = np.concatenate((temp_label, np.reshape([0, data[i].label], (1,2))), axis=0)
	if input_train.size == 0:
		input_train = temp_input
		output_train = temp_label
	else :
		input_train = np.concatenate((input_train, temp_input), axis=0)
		output_train = np.concatenate((output_train, temp_label), axis=0)
if printLog:
	print("input matrix:(train) " + str(input_train.shape))
	print(input_train)
	print("output matrix:(train) " + str(output_train.shape))
	print(output_train)

#handle testing data
testdatas = ptt.handleFile (parsed_test, highlighted_test)
for testdata in testdatas:
	temp_testinput = np.array([])
	temp_testlabel = np.zeros((1, 2))
	for i in testdata:
		if (testdata [i].vector().shape[0]) == 1:
			if temp_testinput.size == 0:
				temp_testinput = testdata [i].vector()
				temp_testlabel[0,:] = [0, testdata[i].label]
				print(temp_testlabel)
				print(str(temp_testlabel.shape))
			else:
				temp_testinput = np.concatenate((temp_testinput, testdata [i].vector()), axis=0)
				temp_testlabel = np.concatenate((temp_testlabel, np.reshape([0, testdata[i].label], (1,2))), axis=0)
	if input_test.size == 0:
		input_test = temp_testinput
		output_test = temp_testlabel
	else :
		input_test = np.concatenate((input_test, temp_testinput), axis=0)
		output_test = np.concatenate((output_test, temp_testlabel), axis=0)
if printLog:
	print("input matrix:(test) " + str(input_test.shape))
	print(input_test)
	print("output matrix:(test) " + str(output_test.shape))
	print(output_test)

features = input_train.astype(np.float32)
labels = output_train.astype(np.float32)

num_output_classes = 2
input_dim = features.shape[1]
output_dim = 2

num_hidden_layers = 100
hidden_layers_dim = 50

input = input_variable(input_dim, np.float32)
label = input_variable((num_output_classes), np.float32)


def linear_layer(input_var, output_dim):

    input_dim = input_var.shape[0]
    times_param = parameter(shape=(input_dim, output_dim))
    bias_param = parameter(shape=(output_dim))

    t = times(input_var, times_param)
    return bias_param + t

def dense_layer(input, output_dim, nonlinearity):
    r = linear_layer(input, output_dim)
    r = nonlinearity(r)
    return r;

# Define a multilayer feedforward classification model
def fully_connected_classifier_net(input, num_output_classes, hidden_layer_dim, 
                                   num_hidden_layers, nonlinearity):
    
    h = dense_layer(input, hidden_layer_dim, nonlinearity)
    for i in range(1, num_hidden_layers):
        h = dense_layer(h, hidden_layer_dim, nonlinearity)
    r = linear_layer(h, num_output_classes)
    return r

# Create the fully connected classfier
z = fully_connected_classifier_net(input, num_output_classes, hidden_layers_dim, 
                                   num_hidden_layers, sigmoid)


loss = cross_entropy_with_softmax(z, label)

eval_error = classification_error(z, label)

# Instantiate the trainer object to drive the model training
learning_rate = 0.3
lr_schedule = learning_rate_schedule(learning_rate, UnitType.minibatch) 
learner = sgd(z.parameters, lr_schedule)
trainer = Trainer(z, loss, eval_error, [learner])

# Define a utility function to compute the moving average sum.
# A more efficient implementation is possible with np.cumsum() function
def moving_average(a, w=10):
    
    if len(a) < w: 
        return a[:]    # Need to send a copy of the array
    return [val if idx < w else sum(a[(idx-w):idx])/w for idx, val in enumerate(a)]


# Defines a utility that prints the training progress
def print_training_progress(trainer, mb, frequency, verbose=1):
    
    training_loss = "NA"
    eval_error = "NA"

    if mb%frequency == 0:
        training_loss = get_train_loss(trainer)
        eval_error = get_train_eval_criterion(trainer)
        if verbose: 
            print ("Minibatch: {}, Train Loss: {}, Train Error: {}".format(mb, training_loss, eval_error))
        
    return mb, training_loss, eval_error

# Initialize the parameters for the trainer
minibatch_size = 25
num_samples = input_train.shape[0]
num_minibatches_to_train = num_samples / minibatch_size

# Run the trainer and perform model training
training_progress_output_freq = 20

plotdata = {"batchsize":[], "loss":[], "error":[]}

for i in range(0, int(num_minibatches_to_train)):
    
    # Specify the input variables mapping in the model to actual minibatch data for training
    trainer.train_minibatch({input : features, label : labels})
    batchsize, loss, error = print_training_progress(trainer, i, 
                                                     training_progress_output_freq, verbose=0)
    
    if not (loss == "NA" or error =="NA"):
        plotdata["batchsize"].append(batchsize)
        plotdata["loss"].append(loss)
        plotdata["error"].append(error)

# Compute the moving average loss to smooth out the noise in SGD

plotdata["avgloss"] = moving_average(plotdata["loss"])
plotdata["avgerror"] = moving_average(plotdata["error"])

test_features = input_test.astype(np.float32)
test_labels = output_test.astype(np.float32)

trainer.test_minibatch({input: test_features, label: test_labels})
out = softmax(z)
predicted_label_prob = out.eval({input: test_features})

print("Label    :", np.argmax(labels[:test_features.shape[0]],axis=1))
print("Predicted:", np.argmax(predicted_label_prob[0,:test_features.shape[0],:],axis=1))
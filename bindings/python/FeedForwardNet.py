# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import numpy as np
import sys
import os
from cntk.device import cpu, set_default_device
from cntk import Trainer
from cntk.learner import sgd
from cntk.ops import input_variable, cross_entropy_with_softmax, classification_error, sigmoid
from cntk.utils import ProgressPrinter

abs_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(abs_path, "..", ".."))
from python.examples.common.nn import fully_connected_classifier_net

# Creates and trains a feedforward classification model

def ffnet(dinput, doutput):
    input_dim = dinput.shape[1]
    num_output_classes = 2
    num_hidden_layers = 2
    hidden_layers_dim = 50

    # Input variables denoting the features and label data
    input = input_variable((input_dim), np.float32)
    label = input_variable((num_output_classes), np.float32)
    print(label)

    # Instantiate the feedforward classification model
    netout = fully_connected_classifier_net(
        input, num_output_classes, hidden_layers_dim, num_hidden_layers, sigmoid)

    ce = cross_entropy_with_softmax(netout, label)
    pe = classification_error(netout, label)

    # Instantiate the trainer object to drive the model training
    trainer = Trainer(netout, ce, pe, sgd(netout.parameters, lr=0.02))

    pp = ProgressPrinter(128)
    features = dinput
    labels = doutput

    trainer.train_minibatch({input: features, label: labels})
    pp.update_with_trainer(trainer)
    pp.epoch_summary()    
    print("Training completed")

    #test_features = testinput
    #test_labels = testoutput
    #avg_error = trainer.test_minibatch(
    #    {input: test_features, label:test_labels})
    return

if __name__ == '__main__':
    # Specify the target device to be used for computing, if you do not want to
    # use the best available one, e.g.
    # set_default_device(cpu())

    error, test_labels = ffnet()
    print(" error rate on an unseen minibatch %f" % error)
    print(test_labels)

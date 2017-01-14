"""
Unit tests for the most trivial case of global conditioning I could think of,
where behavior of the hypernet is trained to be either an OR, AND, NOT, XOR or
IDENTITY logic gate, depending upon a global condition category selecting which
of the above it should behave as.
."""
import sys
import os
import json
import numpy as np
import tensorflow as tf
import random
from wavenet import ParamProducerModel, optimizer_factory
from wavenet import InputSpec, ParamSpec, ParamTree, ComputedParm

#sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# Types are OR, AND, XOR, IDENTITY, NOT
NUM_LOGIC_TYPES = 5
NUM_HIDDEN_UNITS = 32
NUM_INPUTS = 2
NUM_OUTPUTS = 2


'''
A net that we train to act as one of various kinds of logic gates. Its
parameters are produced by an MLP (ParamProducerModel), conditioned upon the
kind of behavior we want it to learn.
'''
class LogicNetModel:
    def __init__(self):
        # Create the parameters specs.
        self.param_specs = self._create_param_specs()
        self.output=None

    def _make_spec(self, name, shape, kind):
        print("ComputedParam:{}".format(name))
        return ComputedParm(name=name, shape=shape, kind=kind)

    def _create_param_specs(self):
        t = ParamTree('LogicNet')
        l = t.add_child('layers')
        c = l.add_child('layer1')
        c.add_param(self._make_spec(name='weights',
                                    shape=[NUM_INPUTS, NUM_HIDDEN_UNITS],
                                    kind='filter'))
        c.add_param(self._make_spec(name='biases',
                                    shape=[NUM_HIDDEN_UNITS],
                                    kind='bias'))

        c = l.add_child('layer2')
        c.add_param(self._make_spec(name='weights',
                                    shape=[NUM_HIDDEN_UNITS, NUM_OUTPUTS],
                                    kind='filter'))
        c.add_param(self._make_spec(name='biases',
                                    shape=[NUM_OUTPUTS],
                                    kind='bias'))
        return t

    def Forward(self, input, parameters):
        params = parameters['LogicNet']['layers']
        w1 = params['layer1']['weights']
        b1 = params['layer1']['biases']
        w2 = params['layer2']['weights']
        b2 = params['layer2']['biases']

        current = tf.matmul(input, w1) + b1
        current = tf.nn.relu(current)
        current = tf.matmul(current, w2) + b2
        current = tf.nn.sigmoid(current)
        return current

    def TrainingLoss(self, input, parameters, target):
        self.output = self.Forward(input, parameters)
        loss = tf.nn.l2_loss(target-self.output)
        return loss


def training_inputs():
    inputs = [ (0.1, 0.1),
               (0.1, 0.9),
               (0.9, 0.1),
               (0.9, 0.9) ]
    return inputs


def superimpose_noise(data):
    new_data = []
    for vals in data:
        noise = np.random.uniform(low=-0.1, high=0.1, size=2)
        new_val = (vals[0] + noise[0], vals[1] + noise[1])
        new_data.append(new_val)
    return new_data


def identity_set():
    outputs = [ (0.1, 0.1),
                (0.1, 0.9),
                (0.9, 0.1),
                (0.9, 0.9) ]
    outputs = superimpose_noise(outputs)
    return outputs


# First output is the result of the OR operation, and second is its
# complement.
def or_set():
    outputs = [ (0.1, 0.9),
                (0.9, 0.1),
                (0.9, 0.1),
                (0.9, 0.1) ]
    outputs = superimpose_noise(outputs)
    return outputs


# First output is the result of the AND operation, and second is its
# complement.
def and_set():
    outputs = [ (0.1, 0.9),
                (0.1, 0.9),
                (0.1, 0.9),
                (0.9, 0.1) ]
    outputs = superimpose_noise(outputs)
    return outputs


def xor_set():
    outputs = [ (0.1, 0.9),
                (0.9, 0.1),
                (0.9, 0.1),
                (0.1, 0.9) ]
    outputs = superimpose_noise(outputs)
    return outputs


def not_set():
    outputs = [ (0.9, 0.9),
                (0.9, 0.1),
                (0.1, 0.9),
                (0.1, 0.1) ]
    outputs = superimpose_noise(outputs)
    return outputs


class TestHyperNet(tf.test.TestCase):
    def setUp(self):
        self.optimizer_type = 'sgd'
        self.learning_rate = 0.001
        self.generate = True
        self.momentum = 0.9
        self.global_conditioning = False
        self.train_iters = 500
        self.net = LogicNetModel()

        input_spec = InputSpec('category', 'gate_type', {'cardinality': 5})

        self.param_producer = ParamProducerModel(
                                  input_spec,
                                  self.net._create_param_specs(),
                                  residual_channels=40)

        self.training_data_generators = { 0: identity_set,
                                          1: or_set,
                                          2: and_set,
                                          3: xor_set,
                                          4: not_set }

    def testTraining(self):
        np.random.seed(42)
        gate_category_placeholder = tf.placeholder(dtype=tf.int32,
                                                   shape=[1,1])
        gate_input_placeholder = tf.placeholder(dtype=tf.float32,
                                                shape=[1,2])
        gate_target_placeholder = tf.placeholder(dtype=tf.float32,
                                                 shape=[1,2])

        parameters = self.param_producer.create_params(
            gate_category_placeholder)

        loss = self.net.TrainingLoss(gate_input_placeholder,
                                     parameters,
                                     gate_target_placeholder)

        optimizer = optimizer_factory['sgd'](learning_rate=self.learning_rate,
                                             momentum=self.momentum)
        trainable = tf.trainable_variables()
        optim = optimizer.minimize(loss, var_list=trainable)
        ops = [loss, optim]
        init = tf.global_variables_initializer()
        with self.test_session() as sess:
            sess.run(init)

            for i in range(self.train_iters):
                # iterate through each of the logic gate types
                for gate_id in range(5):
                    inputs = training_inputs()
                    targets = self.training_data_generators[gate_id]()
                    assert len(inputs) == len(targets)
                    for j in range(len(inputs)):
                        feed_dict = {gate_category_placeholder: [[gate_id]],
                                     gate_input_placeholder: [inputs[j]],
                                     gate_target_placeholder: [targets[j]]}


                        results = sess.run(ops, feed_dict=feed_dict)

                    if i % 10 == 0:
                        print("i: %d, gate:%d, loss: %f" % (i, gate_id,
                              results[0]))


            # Now generate for each of the five gate types.
            for gate_id in range(5):
                inputs = training_inputs()
                targets = self.training_data_generators[gate_id]()
                assert len(inputs) == len(targets)
                outputs = []
                for j in range(len(inputs)):
                    feed_dict = {gate_category_placeholder: [[gate_id]],
                                 gate_input_placeholder: [inputs[j]]}

                    outputs.append(sess.run(
                        self.net.output, feed_dict=feed_dict))

                for (input, output, target) in zip(inputs, outputs,
                                                   targets):
                    self.assertAlmostEqual(output[0][0], target[0], delta=0.2)
                    self.assertAlmostEqual(output[0][1], target[1], delta=0.2)




if __name__ == '__main__':
    tf.test.main()

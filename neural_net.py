"""neural_net.py
~~~~~~~~~~~~~~~~~~~~

Implementing the stochastic gradient descent learning algorithm for 
a feedforward neural network. The main objective of the neural network
is considered to be regression, however, with minor modifications, it
can apply to classification problems as well.

The core of this code is taken from Michael A. Nielsen book
``Neural Networks and Deep Learning``
http://neuralnetworksanddeeplearning.com/
"""

#### Libraries
# Standard library
import json
import random
import sys
import time

# Third-party libraries
import numpy as np
import matplotlib.pyplot as plt



##### Defining the underlying function for regression

class Fn(object):
    """The example function that neural network is going to learn to approximate"""

    def __init__(self, xmin, xmax, fn):
        """
        Define the function fn in the interval [xmin, xmax]. For a 
        multi-dimensional function, ``xmin`` and ``xmax`` are, respectively,
        a list of lower bound and upper bound values. For example, for a 
        2-dimensional function fn=fn(x,y), xmin = [x1, y1] and xmax = [x2, y2] means 
        x in [x1, x2] and y in [y1, y2].
        
        """
        if not hasattr(xmin, "__iter__"):
            xmin = [xmin]
        if not hasattr(xmax, "__iter__"):
            xmax = [xmax]
        assert len(xmin) == len(xmax), "xmin and xmax have to have same length" 
        self.xmin = xmin
        self.xmax = xmax
        self.func = fn


#### Define the quadratic, cross-entropy and "exponential" cost functions

class QuadraticCost(object):

    @staticmethod
    def fn(a, y):
        """Return the cost associated with an output ``a`` and desired output
        ``y``. Note that in our convention the output of the last layer is
        transpose of the last layer activation y_out = a.transpose(). 

        """
        y_out = a.transpose()
        return 0.5*np.linalg.norm(y_out-y)**2

    @staticmethod
    def delta(a_prime, a, y):
        """
        Return the error delta from the output layer.
        Note that in our convention the output of the last layer is
        transpose of the last layer activation y_out = a.transpose().
        ``a_prime`` is da/dz of the output layer. 
        """
        return (a-y.transpose()) * a_prime


class CrossEntropyCost(object):

    @staticmethod
    def fn(a, y):
        """Return the cost associated with an output ``a`` and desired output
        ``y``.  Note that np.nan_to_num is used to ensure numerical
        stability.  In particular, if both ``a`` and ``y`` have a 1.0
        in the same slot, then the expression (1-y)*np.log(1-a)
        returns nan.  The np.nan_to_num ensures that that is converted
        to the correct value (0.0).
        Note that in our convention the output of the last layer is
        transpose of the last layer activation y_out = a.transpose().
        For a classification problem, y=0 or 1 and the cost function at 
        its minimum vanishes. We have substracted the non-zero minimum for
        a general regression problem.

        """
        y_out = a.transpose()
        return np.sum(np.nan_to_num(-y*np.log(y_out)-(1-y)*np.log(1-y_out))) - \
                np.sum(np.nan_to_num(-y*np.log(y)-(1-y)*np.log(1-y)))

    @staticmethod
    def delta(a_prime, a, y):
        """Return the error delta from the output layer.  Note that the
        parameter ``a_prime`` is not used by the method.  It is included in
        the method's parameters in order to make the interface
        consistent with the delta method for other cost classes.
        Note that in our convention the output of the last layer is
        transpose of the last layer activation y_out = a.transpose().

        """
        return (a-y.transpose())/(a*(1-a)) * a_prime
        #return (a-y.transpose())

    
class ExponentialCost(object):
    
    @staticmethod
    def fn(a,y):
        """
        Return the cost associated with an output ``a`` and desired output
        ``y``. This cost function is desirable if the activation of output
        neuron is an exponential function. It avoids slow down of learning
        when the output neuron is saturated. 
        Note that in our convention the output of the last layer is
        transpose of the last layer activation y_out = a.transpose().
        np.nan_to_num is used to ensure numerical stability.
        """
        y_out = a.transpose()
        return np.sum(np.nan_to_num(y_out-y - y*np.log(y_out/y)))
    
    @staticmethod
    def delta(a_prime, a, y):
        """Return the error delta from the output layer.  Note that the
        parameter ``a_prime`` is not used by the method.  It is included in
        the method's parameters in order to make the interface
        consistent with the delta method for other cost classes.
        Note that in our convention the output of the last layer is
        transpose of the last layer activation y_out = a.transpose().
        
        """
        return (1-y.transpose()/a) * a_prime


#### Main Network class
class Network(object):

    def __init__(self, sizes, activations=None, Fn=None, cost=CrossEntropyCost):
        """The list ``sizes`` contains the number of neurons in the respective
        layers of the network.  For example, if the list was [2, 3, 1]
        then it would be a three-layer network, with the first layer
        containing 2 neurons, the second layer 3 neurons, and the
        third layer 1 neuron.  The biases and weights for the network
        are initialized randomly, using
        ``self.default_weight_initializer`` (see docstring for that
        method).
        Fn is the function object, if known, that the neural networks is trying to 
        approximate (see docstring for Fn).
        activations specifies the activation function for each layer of 
        the neural network. If it's not given, all activations are assumed
        to be sigmoid.

        """
        if activations==None:
            activations=["Sigmoid" for j in range(len(sizes)-1)]
        self.num_layers = len(sizes)
        self.sizes = sizes 
        self.default_weight_initializer()
        self.cost=cost
        self.function = Fn
        self.activations = activations
        for a in activations:
            assert a in ["Sigmoid", "Exponential", "RecL"], "%s is not a valid activation" %a
        assert len(activations) == len(sizes)-1, "number of activations and layers do not match"

    def default_weight_initializer(self):
        """Initialize each weight using a Gaussian distribution with mean 0
        and standard deviation 1 over the square root of the number of
        weights connecting to the same neuron.  Initialize the biases
        using a Gaussian distribution with mean 0 and standard
        deviation 1.

        Note that the first layer is assumed to be an input layer, and
        by convention we won't set any biases for those neurons, since
        biases are only ever used in computing the outputs from later
        layers.

        """
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x)/np.sqrt(x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]
    
    def activation(self, z, function):
        """Return the output of activation ``function`` for a given input ``z``."""
        if function=="Sigmoid":
            return sigmoid(z)
        elif function=="Exponential":
            return np.exp(z)
        elif function=="RecL":
            return np.maximum(z, 0)
    
    def activation_prime(self, z, function):
        """Return the derivation of activation ``function`` for a given input ``z``."""
        if function=="Sigmoid":
            return sigmoid_prime(z)
        elif function=="Exponential":
            return np.exp(z)
        elif function=="RecL":
            return 1*(z>=0)
        
    def feedforward(self, X):
        """
        Return the output of the network if ``X`` is input.
        First row in the output corresponds to the first row in X, and so on.
        
        """
        a = X.transpose()
        for b, w, function in zip(self.biases, self.weights, self.activations):
            a = self.activation(np.dot(w, a)+b, function)
        return a.transpose()

    def SGD(self, training_data, mini_batch_size, eta, epochs=50, mu=0.0,
            no_improvement_size=50,
            variable_learning=True,
            learning_halve=5,
            lmbda = 0.0,
            evaluation_data=None,
            monitor_evaluation_cost=True,
            monitor_evaluation_accuracy=True,
            monitor_training_cost=True,
            monitor_training_accuracy=True,
            show_plot=True,
            print_epoch=True):
        """Train the neural network using mini-batch stochastic gradient
        descent.  The ``training_data`` is a tuple ``(X, y)``. 
        X is a n*n_in numpy array representing "n" input data,
        each with "n_in" feature (n_in is the number of neurons in the input layer.)
        Each row of X represents one input data. y is a n*n_out numpy array
        representing the corresponding "n" desired output (n_out is the 
        number of neurons in the output layer.) Each row in y is one output.
        Note: In some conventions, the definition of X and y are different from
        ours by a transpose.
        The other non-optional parameters are self-explanatory, as is the
        regularization parameter ``lmbda``. ``mu`` is the momentum coefficient,
        where mu=0 corresponds to normal stochastic gradient descent with no
        momentum contribution. The value of ``mu`` has to be chosen between
        0 and 1 where 1 corresponds to no friction term.
        The method also accepts ``evaluation_data``, usually either 
        the validation or test data.  We can monitor the cost and accuracy 
        on either the evaluation data or the training data, by setting the
        appropriate flags.
        All values are evaluated at the end of each training epoch.  So, for example,
        if we train for 30 epochs, then the first element of the tuple
        will be a 30-element list containing the cost on the
        evaluation data at the end of each epoch. Note that the lists
        are empty if the corresponding flag is not set.
        If show_plot flag is True, at the end of training, it plots 
        the cost/accuracy (per epoch) for True monitoring flags.
        Set print_epoch=True if you want to monitor the speed 
        of learning program.
        """
        X, y = training_data
        n = len(X)
        evaluation_cost, evaluation_accuracy = [], []
        training_cost, training_accuracy = [], []
        random_order = list(range(n))
        # velocities associated to weights and biases
        v_weights = [np.zeros(w.shape) for w in self.weights]
        v_biases = [np.zeros(b.shape) for b in self.biases]
        epoch = 1
        # record of the best optimization point in gradient descent
        best_cost = self.total_cost(training_data, lmbda)
        best_weights = self.weights[:]
        best_biases = self.biases[:]
        best_epoch = epochs
        best_v_weights = v_weights[:]
        best_v_biases = v_biases[:]
        last_improvement = 0
        # for variable_learning=True, we keep a record of the epoch
        # where learning rate is halved to represents in the cost and 
        # accuracy plots.
        epoch_threshold = []
        start_time = time.time()
        while epoch <= epochs or variable_learning:
            # randomly ordering the training data and putting them into
            # mini batches.
            random.shuffle(random_order)
            mini_batches = [(X[random_order[k:k+mini_batch_size]],
                            y[random_order[k:k+mini_batch_size]])
                            for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                v_weights, v_biases = self.update_mini_batch(
                    mini_batch, eta, mu, lmbda, n, v_weights, v_biases)    
            if print_epoch and time.time()-start_time > 20.0:
                start_time = time.time()
                print("Epoch %s training complete." % epoch, "learning_halve = %s" %learning_halve)
            epoch += 1
            if monitor_training_cost:
                cost = self.total_cost(training_data, lmbda)
                training_cost.append(cost)
            if monitor_training_accuracy:
                accuracy = self.accuracy(training_data)
                training_accuracy.append(accuracy)
            if monitor_evaluation_cost:
                cost = self.total_cost(evaluation_data, lmbda)
                evaluation_cost.append(cost)
            if monitor_evaluation_accuracy:
                accuracy = self.accuracy(evaluation_data)
                evaluation_accuracy.append(accuracy)
            if variable_learning:
                # if a better point in optimization is found, make 
                # a copy of that instant. 
                if self.total_cost(training_data, lmbda) < best_cost:
                    best_cost = self.total_cost(training_data, lmbda)
                    best_weights = self.weights[:]
                    best_biases = self.biases[:]
                    best_epoch = epoch
                    best_v_weights = v_weights[:]
                    best_v_biases = v_biases[:]
                    last_improvement = 0
                else:
                    last_improvement += 1
                # if no improvement is made within no_improvement_size,
                # halve the learning rate and start from the best optimization point
                if last_improvement == no_improvement_size:
                    learning_halve -= 1
                    epoch_threshold.append(epoch)
                    last_improvement = 0
                    eta = eta/2
                    self.weights = best_weights[:]
                    self.biases = best_biases[:]
                    v_weights = best_v_weights[:]
                    v_biases = best_v_biases[:]
                if learning_halve == 0:
                    self.weights = best_weights
                    self.biases = best_biases
                    break
        if show_plot:
            plt.figure()
            iteration = list(range(1, epoch))
            if monitor_training_cost:
                plt.plot(iteration, training_cost, label="training cost")
            if monitor_evaluation_cost:
                plt.plot(iteration, evaluation_cost, label="evaluation cost")
            plt.legend()
            plt.xlabel("epoch")
            plt.ylabel("cost")
            plt.xscale("log")
            plt.yscale("log")
            for threshold in epoch_threshold:
                plt.axvline(x=threshold, color="black", linestyle="dashed")
            plt.title("cost vs number of iterations, best epoch=%s" % best_epoch)

            plt.figure()
            if monitor_training_accuracy:
                plt.plot(iteration, training_accuracy, label="training accuracy")
            if monitor_evaluation_accuracy:
                plt.plot(iteration, evaluation_accuracy, label="evaluation accuracy")
            plt.legend()
            plt.xlabel("epoch")
            plt.ylabel("accuracy")
            plt.xscale("log")
            plt.yscale("log")
            for threshold in epoch_threshold:
                plt.axvline(x=threshold, color="black", linestyle="dashed")
            plt.title("accuracy vs number of iterations")

    def update_mini_batch(self, mini_batch, eta, mu, lmbda, n, v_weights, v_biases):
        """Update the network's weights and biases by applying gradient
        descent using backpropagation to a single mini batch.  The
        ``mini_batch`` is a tuple ``(X, y)``, ``eta`` is the
        learning rate, ``lmbda`` is the regularization parameter, and
        ``n`` is the total number of the training data set.

        """
        X, y = mini_batch
        nabla_b, nabla_w = self.backprop(X, y)
        v_weights = [mu*v_w-eta*(lmbda/n)*w-(eta/len(X))*nw
                    for v_w, w, nw in zip(v_weights, self.weights, nabla_w)]
        v_biases = [mu*v_b-(eta/len(X))*nb
                   for v_b, nb in zip(v_biases, nabla_b)]
        self.weights = [w + v_w
                        for w, v_w in zip(self.weights, v_weights)]
        self.biases = [b + v_b
                       for b, v_b in zip(self.biases, v_biases)]
        return v_weights, v_biases

    def backprop(self, X, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_X.  if X has multiple rows
        (multiple inputs), it returns the sum of the cost function gradient
        for inputs in X. ``nabla_b`` and ``nabla_w`` are layer-by-layer 
        lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = X.transpose()
        activations = [X.transpose()] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w, function in zip(self.biases, self.weights, self.activations):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = self.activation(z, function)
            activations.append(activation)
        # backward pass
        delta = self.cost.delta(self.activation_prime(zs[-1], self.activations[-1])
                                , activations[-1], y)
        nabla_b[-1] = np.sum(delta, axis=1).reshape(len(delta), 1)
        nw = np.dot(delta, activations[-2].transpose())
        nabla_w[-1] = np.sum(nw, axis=1).reshape(len(delta),1)
        # Here, l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = self.activation_prime(z, self.activations[-l])
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = np.sum(delta, axis=1).reshape(len(delta), 1)
            nw = np.dot(delta, activations[-l-1].transpose())
            nabla_w[-l] = np.sum(nw, axis=1).reshape(len(delta),1)
        return nabla_b, nabla_w

    def accuracy(self, data, flag="MeanSquared"):
        """
        Return the error between the prediction of neural net and the
        desired value. If flag="MeanSquared" returns the mean squared error.
        If flag="Max", returns the maximum value of error.
        
        """
        X, y = data
        error = np.linalg.norm(self.feedforward(X)-y, axis=1)
        if flag=="MeanSquared":
            return np.sqrt(np.sum(error**2)/len(error))
        elif flag=="Max":
            return np.max(error)

    def total_cost(self, data, lmbda):
        """Return the total cost for the data set ``data``.
        """
        X, y = data
        activation_out = self.feedforward(X).transpose()
        cost = self.cost.fn(activation_out, y)/len(X)
        cost += 0.5*(lmbda/len(X))*sum(
            np.linalg.norm(w)**2 for w in self.weights)
        return cost

    def save(self, filename):
        """Save the neural network to the file ``filename``."""
        data = {"sizes": self.sizes,
                "weights": [w.tolist() for w in self.weights],
                "biases": [b.tolist() for b in self.biases],
                "cost": str(self.cost.__name__)}
        f = open(filename, "w")
        json.dump(data, f)
        f.close()

    def plot(self, res=0.01):
        """
        Plots a figure showing the approximation of neural net
        of self.function and the function itself. This method assumes 
        that the network has 1 input.
        """
        t = np.arange(self.function.xmin[0], self.function.xmax[0], res)
        t = t.reshape(len(t), 1)
        y = self.feedforward(t)
        plt.figure()
        plt.plot(t, y, label="net prediction")
        plt.plot(t, self.function.func(t), label="desired function")
        plt.legend(loc="upper center")
        plt.title("network size: {0}".format(self.sizes))

    def plot_hidden(self, res=0.01):
        """
        Plots the input and output values of neurons in the net.
        It produces L-1 (L:number of layers) figures. Each figure
        containts 2 plots, input and output to the layer, from 2nd layer
        onwards. This method assumes that the network has 1 input.
        """
        t = np.arange(self.function.xmin[0], self.function.xmax[0], res)
        a = t.reshape(1, len(t))
        for l in range(len(self.sizes)-1):
            w = self.weights[l]
            b = self.biases[l]
            func = self.activations[l]
            z = np.dot(w, a) + b
            plt.figure()
            plt.subplot(1,2,1)
            for neuron_input in z:
                plt.plot(t, neuron_input)
            plt.title("weighted inputs into layer %s" % (l+1))
            a = self.activation(z, func)
            plt.subplot(1,2,2)
            for neuron_output in a:
                plt.plot(t, neuron_output)
            plt.title("activations of layer %s" % (l+1))



#### Loading a Network

def load(filename):
    """Load a neural network from the file ``filename``.  Returns an
    instance of Network.

    """
    f = open(filename, "r")
    data = json.load(f)
    f.close()
    cost = getattr(sys.modules[__name__], data["cost"])
    net = Network(data["sizes"], cost=cost)
    net.weights = [np.array(w) for w in data["weights"]]
    net.biases = [np.array(b) for b in data["biases"]]
    return net


#### producing data for learning

def produce_data(n, Fn):
    """Uniformly take n samples from Fn object and returns a tuple (X, y).
    Each row of X (x) is a sample and the corresponding row in y has the desired value Fn.fn(x).
    
    """
    X = []
    for xmin, xmax in zip(Fn.xmin, Fn.xmax):
        x = np.random.uniform(xmin, xmax, n)
        X.append(x)
    X = np.array(X).transpose()
    y = []
    for x in X:
        y.append(Fn.func(*x))
    y = np.array(y)
    y = y.reshape(n, y.size//n)
    return (X, y)


def load_data(training_n, validation_n, test_n, Fn):
    """Returns training, cross validation and test data lists with the given format
    specified in produce_data of the function Fn. Sizes of the lists are fixed by the input."""
    return produce_data(training_n, Fn), produce_data(validation_n, Fn), produce_data(test_n, Fn)


def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))

"""
network.py
~~~~~~~~~~

A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network.  Gradients are calculated
using backpropagation.  Note that I have focused on making the code
simple, easily readable, and easily modifiable.  It is not optimized,
and omits many desirable features.
"""

#### Libraries
# Standard library
import random

# Third-party libraries
import numpy as np

class Network(object):
    """
    A feedforward neural network class.

    This class implements a fully-connected (dense) neural network where
    each neuron in one layer connects to every neuron in the next layer.
    The network uses:
    - Sigmoid activation functions for all neurons
    - Quadratic cost function (mean squared error)
    - Stochastic gradient descent with mini-batches for training
    - Backpropagation for computing gradients
    """

    def __init__(self, sizes):
        """
        Initialize the neural network.

        Parameters
        ----------
        sizes : list of int
            The number of neurons in each layer of the network.
            For example, [784, 30, 10] creates a 3-layer network:
            - Input layer: 784 neurons (e.g., 28x28 pixel image)
            - Hidden layer: 30 neurons
            - Output layer: 10 neurons (e.g., digits 0-9)

        Notes
        -----
        - Weights and biases are initialized from a standard normal distribution
          (Gaussian with mean=0, variance=1)
        - The input layer has no biases (biases only affect computation in
          subsequent layers)
        - Weight matrix dimensions: weights[l] has shape (neurons_in_layer_l+1, neurons_in_layer_l)
          This allows computing: activation = sigmoid(weights @ input + bias)
        """
        # Store the network architecture
        self.num_layers = len(sizes)
        self.sizes = sizes

        # Debug: Print the network architecture details
        print("sizes:", sizes)
        print("sizes[1:]:", sizes[1:])  # All layers except input (for biases)
        print("sizes[:-1]:", sizes[:-1])  # All layers except output (for weight connections)
        print("zip(sizes[:-1], sizes[1:]):", list(zip(sizes[:-1], sizes[1:])))

        # Initialize biases for each layer (except input layer)
        # biases[l] is a column vector of shape (neurons_in_layer_l+1, 1)
        # Example: sizes=[784,30,10] -> biases shapes: [(30,1), (10,1)]
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]

        # Initialize weights connecting each pair of adjacent layers
        # weights[l] connects layer l to layer l+1
        # Shape: (neurons_in_next_layer, neurons_in_current_layer)
        # Example: sizes=[784,30,10] -> weights shapes: [(30,784), (10,30)]
        # This allows: next_activation = sigmoid(weights[l] @ current_activation + biases[l])
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

        # Debug: Print initialization summary
        print("Network initialized with {} layers.".format(self.num_layers))
        print("Biases shape:", [b.shape for b in self.biases])
        print("Weights shape:", [w.shape for w in self.weights])

    def feedforward(self, a):
        """
        Compute the output of the network for a given input.

        Parameters
        ----------
        a : numpy.ndarray
            Input vector of shape (n_input, 1) where n_input is the
            number of neurons in the input layer.

        Returns
        -------
        numpy.ndarray
            Output vector of shape (n_output, 1) containing the
            activation values of the output layer neurons.

        Notes
        -----
        For each layer l, computes: a^(l+1) = sigmoid(w^l @ a^l + b^l)
        where @ denotes matrix multiplication.
        """
        for b, w in zip(self.biases, self.weights):
            # Compute weighted sum plus bias, then apply sigmoid activation
            # w @ a gives the weighted sum: shape (n_next, n_current) @ (n_current, 1) = (n_next, 1)
            # Adding b (shape n_next, 1) gives the pre-activation (z)
            # sigmoid squashes the output to range (0, 1)
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        """
        Train the neural network using mini-batch Stochastic Gradient Descent.

        Parameters
        ----------
        training_data : list of tuples
            A list of (x, y) tuples where x is the input and y is the
            desired output (label). Both are numpy arrays.
        epochs : int
            Number of complete passes through the training dataset.
        mini_batch_size : int
            Number of training examples in each mini-batch.
            Smaller batches = noisier gradients but more frequent updates.
            Larger batches = smoother gradients but slower convergence.
        eta : float
            Learning rate - controls the step size in gradient descent.
            Too high: may overshoot minima and diverge.
            Too low: very slow convergence.
        test_data : list of tuples, optional
            If provided, the network will be evaluated on this data
            after each epoch to track learning progress.

        Notes
        -----
        SGD works by:
        1. Shuffling training data (ensures random mini-batch composition)
        2. Dividing into mini-batches
        3. For each mini-batch, computing gradients via backpropagation
        4. Updating weights and biases in the direction that reduces cost

        The update rule is: θ = θ - (η/m) * Σ∇C_x
        where θ represents weights/biases, η is learning rate,
        m is mini-batch size, and ∇C_x is the gradient for sample x.
        """
        # Get the size of test data if provided (for progress reporting)
        if test_data: n_test = len(test_data)
        n = len(training_data)

        # Main training loop - iterate through epochs
        for j in xrange(epochs):
            # Shuffle training data to ensure random mini-batch composition
            # This helps prevent the network from learning order-dependent patterns
            random.shuffle(training_data)

            # Partition training data into mini-batches
            # Each mini-batch is a list of (x, y) tuples
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in xrange(0, n, mini_batch_size)]

            # Update weights and biases for each mini-batch
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)

            # Report progress after each epoch
            if test_data:
                # If test data provided, show accuracy
                print "Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(test_data), n_test)
            else:
                # Otherwise, just indicate epoch completion
                print "Epoch {0} complete".format(j)

    def update_mini_batch(self, mini_batch, eta):
        """
        Update weights and biases using gradient descent on a single mini-batch.

        Parameters
        ----------
        mini_batch : list of tuples
            A list of (x, y) training examples to compute gradients from.
        eta : float
            Learning rate for gradient descent.

        Notes
        -----
        This method:
        1. Initializes gradient accumulators to zero
        2. For each training example, computes gradients via backpropagation
        3. Sums up all gradients
        4. Updates weights and biases using averaged gradients

        The gradient descent update rule:
        w_new = w_old - (η/m) * Σ(∂C/∂w)
        b_new = b_old - (η/m) * Σ(∂C/∂b)

        Dividing by mini-batch size (m) averages the gradients, making
        the learning rate independent of batch size.
        """
        # Initialize gradient accumulators with zeros (same shape as biases/weights)
        # nabla_b and nabla_w will store the sum of gradients over the mini-batch
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # Compute and accumulate gradients for each training example
        for x, y in mini_batch:
            # Backpropagation returns gradients for this single example
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)

            # Accumulate bias gradients: sum across all examples in mini-batch
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]

            # Accumulate weight gradients: sum across all examples in mini-batch
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        # Gradient descent update
        # new_weight = old_weight - (learning_rate / batch_size) × summed_gradient
        # Dividing by len(mini_batch) averages the gradient over the batch
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """
        Compute gradients using the backpropagation algorithm.

        Parameters
        ----------
        x : numpy.ndarray
            Input vector of shape (n_input, 1).
        y : numpy.ndarray
            Desired output vector of shape (n_output, 1).

        Returns
        -------
        tuple of (nabla_b, nabla_w)
            nabla_b : list of numpy.ndarray
                Gradients of cost with respect to biases for each layer.
            nabla_w : list of numpy.ndarray
                Gradients of cost with respect to weights for each layer.

        Notes
        -----
        Backpropagation computes ∂C/∂w and ∂C/∂b using the chain rule.

        Key equations (where L = last layer, l = current layer):
        1. Output layer error: δ^L = ∇_a C ⊙ σ'(z^L)
           - ∇_a C is the derivative of cost w.r.t. output activations
           - σ'(z^L) is sigmoid derivative at the output layer
           - ⊙ denotes element-wise multiplication

        2. Error propagation: δ^l = ((w^(l+1))^T @ δ^(l+1)) ⊙ σ'(z^l)
           - Errors propagate backward through transposed weights

        3. Bias gradient: ∂C/∂b^l = δ^l
           - Bias gradient equals the error at that layer

        4. Weight gradient: ∂C/∂w^l = δ^l @ (a^(l-1))^T
           - Weight gradient is outer product of error and input activation
        """
        # Initialize gradient storage (same structure as weights and biases)
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # ===== FORWARD PASS =====
        # Store activations and pre-activations (z values) for use in backward pass
        activation = x  # Current layer's activation, starting with input
        activations = [x]  # List to store all activations, layer by layer
        zs = []  # List to store all z vectors (weighted sums before activation)

        # Propagate forward through each layer
        for b, w in zip(self.biases, self.weights):
            # Compute weighted sum: z = w @ a + b
            z = np.dot(w, activation)+b
            zs.append(z)  # Store for backward pass

            # Apply sigmoid activation: a = σ(z)
            activation = sigmoid(z)
            activations.append(activation)  # Store for backward pass

        # ===== BACKWARD PASS =====
        # Compute error at output layer using the quadratic cost function
        # δ^L = (a^L - y) ⊙ σ'(z^L)
        # - cost_derivative returns (a^L - y), the derivative of quadratic cost
        # - sigmoid_prime(z^L) is the derivative of the activation function
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])

        # Store gradients for output layer
        # ∂C/∂b^L = δ^L (bias gradient equals the error)
        nabla_b[-1] = delta
        # ∂C/∂w^L = δ^L @ (a^(L-1))^T (outer product of error and previous activation)
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        # Propagate error backward through hidden layers
        # Note: l = 1 means the last layer, l = 2 is second-to-last, etc.
        # This uses Python's negative indexing for convenience
        for l in xrange(2, self.num_layers):
            # Get pre-activation values for this layer
            z = zs[-l]

            # Compute sigmoid derivative at this layer's z values
            sp = sigmoid_prime(z)

            # Compute error for this layer using backpropagation equation:
            # δ^l = ((w^(l+1))^T @ δ^(l+1)) ⊙ σ'(z^l)
            # - Transpose of next layer's weights propagates the error backward
            # - Element-wise multiply with sigmoid derivative
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp

            # Store gradients for this layer
            nabla_b[-l] = delta  # ∂C/∂b^l = δ^l
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())  # ∂C/∂w^l

        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """
        Evaluate the network's accuracy on test data.

        Parameters
        ----------
        test_data : list of tuples
            A list of (x, y) tuples where x is the input and y is the
            correct label (as an integer, not one-hot encoded).

        Returns
        -------
        int
            Number of test inputs correctly classified.

        Notes
        -----
        Classification is determined by finding the neuron with the
        highest activation in the output layer (argmax).
        """
        # For each test example, compare network's prediction with actual label
        # Network prediction = index of output neuron with highest activation
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]

        # Count correct predictions
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """
        Compute the derivative of the quadratic cost function.

        Parameters
        ----------
        output_activations : numpy.ndarray
            The actual output activations from the network (a^L).
        y : numpy.ndarray
            The desired/target output values.

        Returns
        -------
        numpy.ndarray
            Vector of partial derivatives ∂C/∂a^L.

        Notes
        -----
        For the quadratic cost function C = (1/2)||y - a||^2,
        the derivative with respect to output activations is simply (a - y).
        This is used in the backpropagation algorithm to compute the
        initial error at the output layer.
        """
        return (output_activations-y)


#### Miscellaneous functions

def sigmoid(z):
    """
    The sigmoid (logistic) activation function.

    Parameters
    ----------
    z : numpy.ndarray or float
        The input value(s) - can be a scalar, vector, or matrix.

    Returns
    -------
    numpy.ndarray or float
        The sigmoid of z, with values in the range (0, 1).

    Notes
    -----
    The sigmoid function is defined as: σ(z) = 1 / (1 + e^(-z))

    Properties:
    - Output range: (0, 1) - useful for representing probabilities
    - σ(0) = 0.5
    - Symmetric: σ(-z) = 1 - σ(z)
    - Smooth and differentiable everywhere
    - Can cause vanishing gradients for very large or small z values
    """
    return 1.0/(1.0+np.exp(-z))


def sigmoid_prime(z):
    """
    Compute the derivative of the sigmoid function.

    Parameters
    ----------
    z : numpy.ndarray or float
        The input value(s) at which to compute the derivative.

    Returns
    -------
    numpy.ndarray or float
        The derivative of sigmoid at z.

    Notes
    -----
    The derivative of sigmoid has a convenient form:
    σ'(z) = σ(z) * (1 - σ(z))

    This can be derived from the definition:
    σ'(z) = d/dz [1/(1+e^(-z))] = e^(-z)/(1+e^(-z))^2 = σ(z)*(1-σ(z))

    Properties:
    - Maximum value of 0.25 at z=0
    - Approaches 0 as |z| increases (causes vanishing gradient problem)
    - Always positive
    """
    return sigmoid(z)*(1-sigmoid(z))

from neural_net import NeuralNetwork, NetworkFramework
from neural_net import Node, Target, Input
import random
import math


# <--- Problem 3, Question 1 --->

def FeedForward(network, input):
  """
  Arguments:
  ---------
  network : a NeuralNetwork instance
  input   : an Input instance

  Returns:
  --------
  Nothing

  Description:
  -----------
  This function propagates the inputs through the network. That is,
  it modifies the *raw_value* and *transformed_value* attributes of the
  nodes in the network, starting from the input nodes.

  Notes:
  -----
  The *input* arguments is an instance of Input, and contains just one
  attribute, *values*, which is a list of pixel values. The list is the
  same length as the number of input nodes in the network.

  i.e: len(input.values) == len(network.inputs)

  This is a distributed input encoding (see lecture notes 7 for more
  informations on encoding)

  In particular, you should initialize the input nodes using these input
  values:

  network.inputs[i].raw_value = input[i]
  """
  '''
  Note: We assume that nodes in network.inputs, network.hidden_nodes,
        and network.outputs are sorted in topological order, so we may
        iterate through each list in order and have the invariant
        that all my parents have already been processed.
  '''

  def propagate_forward(nodes):
    for node in nodes:
      node.raw_value = NeuralNetwork.ComputeRawValue(node)
      node.transformed_value = NeuralNetwork.Sigmoid(node.raw_value)

  network.CheckComplete()

  # 1) Assign input values to input nodes
  for i, input_node in enumerate(network.inputs):
    input_node.raw_value = input.values[i]
    # for input nodes, the transformed value is just the raw input value
    input_node.transformed_value = input_node.raw_value

  # 2) Propagates to hidden layer
  # 3) Propagates to the output layer
  propagate_forward(network.hidden_nodes + network.outputs)

#< --- Problem 3, Question 2

def Backprop(network, input, target, learning_rate):
  """
  Arguments:
  ---------
  network       : a NeuralNetwork instance
  input         : an Input instance
  target        : a target instance
  learning_rate : the learning rate (a float)

  Returns:
  -------
  Nothing

  Description:
  -----------
  The function first propagates the inputs through the network
  using the Feedforward function, then backtracks and update the
  weights.

  Notes:
  ------
  The remarks made for *FeedForward* hold here too.

  The *target* argument is an instance of the class *Target* and
  has one attribute, *values*, which has the same length as the
  number of output nodes in the network.

  i.e: len(target.values) == len(network.outputs)

  In the distributed output encoding scenario, the target.values
  list has 10 elements.

  When computing the error of the output node, you should consider
  that for each output node, the target (that is, the true output)
  is target[i], and the predicted output is network.outputs[i].transformed_value.
  In particular, the error should be a function of:

  target[i] - network.outputs[i].transformed_value

  """
  # sets the delta for each node (hidden or output)
  def propagate_backward(nodes):
    for i, node in enumerate(nodes):
      # node is an output node
      if not node.forward_neighbors:
        node.error = target.values[i] - node.transformed_value
      else:
        # only works if we process in topological order, which we assume
        node.error = sum(map(
          lambda (weight, child): weight.value * child.delta,
          zip(node.forward_weights, node.forward_neighbors)
        ))
      node.delta = node.error * NeuralNetwork.SigmoidPrime(node.raw_value)

  """
  updates weights of child from the parent. done a second step separate from
  error/delta calculation so we don't accidently use the next time step's weight
  in our delta calculation
  """
  # def update_weights(nodes):
  #   for node in nodes:
  #     for weight in node.weights:
  #       weight.value += learning_rate * node.transformed_value * node.delta
  def update_weights(nodes):
    for node in nodes:
      for weight, child in zip(node.forward_weights, node.forward_neighbors):
        weight.value += learning_rate * node.transformed_value * child.delta

  network.CheckComplete()
  # 1) We first propagate the input through the network
  FeedForward(network, input)

  # 2) Then we compute the errors and update the weigths starting with the last layer
  # 3) We now propagate the errors to the hidden layer, and update the weights there too
  propagate_backward(network.outputs + network.hidden_nodes[::-1])
  update_weights(network.inputs + network.hidden_nodes)

# <--- Problem 3, Question 3 --->

def Train(network, inputs, targets, learning_rate, epochs):
  """
  Arguments:
  ---------
  network       : a NeuralNetwork instance
  inputs        : a list of Input instances
  targets       : a list of Target instances
  learning_rate : a learning_rate (a float)
  epochs        : a number of epochs (an integer)

  Returns:
  -------
  Nothing

  Description:
  -----------
  This function should train the network for a given number of epochs. That is,
  run the *Backprop* over the training set *epochs*-times
  """
  network.CheckComplete()

  for e in range(epochs):
    for input, target in zip(inputs, targets):
      Backprop(network, input, target, learning_rate)

# <--- Problem 3, Question 4 --->

class EncodedNetworkFramework(NetworkFramework):
  def __init__(self):
    """
    Initializatio.
    YOU DO NOT NEED TO MODIFY THIS __init__ method
    """
    super(EncodedNetworkFramework, self).__init__() # < Don't remove this line >

  # <--- Fill in the methods below --->

  def EncodeLabel(self, label):
    """
    Arguments:
    ---------
    label: a number between 0 and 9

    Returns:
    ---------
    a list of length 10 representing the distributed
    encoding of the output.

    Description:
    -----------
    Computes the distributed encoding of a given label.

    Example:
    -------
    0 => [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    3 => [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    Notes:
    ----
    Make sure that the elements of the encoding are floats.

    """
    # Code seems to expect a Target instance rather than a simple list
    # encoded_label = [0.0] * 10
    # encoded_label[label] = 1.0
    # return encoded_label

    new_target = Target()
    new_target.values = [0.0] * 10
    new_target.values[label] = 1.0
    return new_target


  def GetNetworkLabel(self):
    """
    Arguments:
    ---------
    Nothing

    Returns:
    -------
    the 'best matching' label corresponding to the current output encoding

    Description:
    -----------
    The function looks for the transformed_value of each output, then decides
    which label to attribute to this list of outputs. The idea is to 'line up'
    the outputs, and consider that the label is the index of the output with the
    highest *transformed_value* attribute

    Example:
    -------

    # Imagine that we have:
    map(lambda node: node.transformed_value, self.network.outputs) => [0.2, 0.1, 0.01, 0.7, 0.23, 0.31, 0, 0, 0, 0.1, 0]

    # Then the returned value (i.e, the label) should be the index of the item 0.7,
    # which is 3

    """
    labels = map(lambda node: node.transformed_value, self.network.outputs)
    return labels.index(max(labels))

  def Convert(self, image):
    """
    Arguments:
    ---------
    image: an Image instance

    Returns:
    -------
    an instance of Input

    Description:
    -----------
    The *image* arguments has 2 attributes: *label* which indicates
    the digit represented by the image, and *pixels* a matrix 14 x 14
    represented by a list (first list is the first row, second list the
    second row, ... ), containing numbers whose values are comprised
    between 0 and 256.0. The function transforms this into a unique list
    of 14 x 14 items, with normalized values (that is, the maximum possible
    value should be 1).

    """
    # flatten matrix into list
    new_input = Input()
    new_input.values = [pixel/256.0 for row in image.pixels for pixel in row]
    return new_input

  def InitializeWeights(self):
    """
    Arguments:
    ---------
    Nothing

    Returns:
    -------
    Nothing

    Description:
    -----------
    Initializes the weights with random values between [-0.01, 0.01].

    Hint:
    -----
    Consider the *random* module. You may use the the *weights* attribute
    of self.network.

    """
    for weight in self.network.weights:
        weight.value = random.uniform(-0.01, 0.01)

#<--- Problem 3, Question 6 --->

class SimpleNetwork(EncodedNetworkFramework):
  def __init__(self):
    """
    Arguments:
    ---------
    Nothing

    Returns:
    -------
    Nothing

    Description:
    -----------
    Initializes a simple network, with 196 input nodes,
    10 output nodes, and NO hidden nodes. Each input node
    should be connected to every output node.
    """
    super(SimpleNetwork, self).__init__() # < Don't remove this line >

    # 1) Adds an input node for each pixel.
    for i in range(196):
      new_input = Node()
      self.network.AddNode(new_input, NeuralNetwork.INPUT)

    # 2) Add an output node for each possible digit label.
    for i in range(10):
      new_output = Node()
      self.network.AddNode(new_output, NeuralNetwork.OUTPUT)
      for input_node in self.network.inputs:
        new_output.AddInput(input_node, None, self.network)

#<---- Problem 3, Question 7 --->

class HiddenNetwork(EncodedNetworkFramework):
  def __init__(self, number_of_hidden_nodes=30):
    """
    Arguments:
    ---------
    number_of_hidden_nodes : the number of hidden nodes to create (an integer)

    Returns:
    -------
    Nothing

    Description:
    -----------
    Initializes a network with a hidden layer. The network
    should have 196 input nodes, the specified number of
    hidden nodes, and 10 output nodes. The network should be,
    again, fully connected. That is, each input node is connected
    to every hidden node, and each hidden_node is connected to
    every output node.
    """
    super(HiddenNetwork, self).__init__() # < Don't remove this line >

    # 1) Adds an input node for each pixel
    for i in range(196):
      new_input = Node()
      self.network.AddNode(new_input, NeuralNetwork.INPUT)
    # 2) Adds the hidden layer
    for i in range(number_of_hidden_nodes):
      new_hidden = Node()
      self.network.AddNode(new_hidden, NeuralNetwork.HIDDEN)
      for input_node in self.network.inputs:
        new_hidden.AddInput(input_node, None, self.network)
    # 3) Adds an output node for each possible digit label.
    for i in range(10):
      new_output = Node()
      self.network.AddNode(new_output, NeuralNetwork.OUTPUT)
      for hidden_node in self.network.hidden_nodes:
        new_output.AddInput(hidden_node, None, self.network)


#<--- Problem 3, Question 8 --->

class CustomNetwork(EncodedNetworkFramework):
  def __init__(self, number_of_hidden_nodes=15, learning_rate = .1, edge_probability = 1.0):
    """
    Arguments:
    ---------

    number_of_hidden_nodes : the number of hidden nodes on one layer
    learning_rate : the learning rate that we start at. this decays every time we enter a new epoch
    edge_probability : probability that a node will have an edge with its parent node

    Returns:
    --------
    Nothing

    Description:
    -----------
    Our custom network uses two new strategies for generating hypothesis.
    Firstly, it uses a decaying learning_rate. After every epoch, we decrease
    the learning rate. We give it a lower bound of .0001 Secondly, every time
    we generate a new hidden node our output node, we set a probability p that
    it is connected to any one of its potential parent nodes. We no longer
    have a connected graph, but by reducing the complexity we can potentially
    reduce overfitting.

    """
    super(CustomNetwork, self).__init__() # <Don't remove this line>

    self.learning_rate = learning_rate

    # every epoch, we reduce the learning rate
    def DecayingLRateTrain(network, inputs, targets, learning_rate, epochs):
      network.CheckComplete()
      for e in range(epochs):
        for input, target in zip(inputs, targets):
          Backprop(network, input, target, self.learning_rate)
        self.learning_rate = max(self.learning_rate - (self.learning_rate/25.), .0001)

    # new training function uses decaying learning rate
    self.RegisterTrainFunction(DecayingLRateTrain)

    # 1) Adds an input node for each pixel
    for i in range(196):
      new_input = Node()
      self.network.AddNode(new_input, NeuralNetwork.INPUT)
    # 2) Adds the hidden layer
    for i in range(number_of_hidden_nodes):
      new_hidden = Node()
      self.network.AddNode(new_hidden, NeuralNetwork.HIDDEN)
      for input_node in self.network.inputs:
        if random.random() < .9:
            new_hidden.AddInput(input_node, None, self.network)
    # 3) Adds an output node for each possible digit label.
    for i in range(10):
      new_output = Node()
      self.network.AddNode(new_output, NeuralNetwork.OUTPUT)
      for hidden_node in self.network.hidden_nodes:
        if random.random() < .9:
            new_output.AddInput(hidden_node, None, self.network)

from data_reader import *
from neural_net import *
from neural_net_impl import *
import sys
import random
import matplotlib.pyplot as plt


def parseArgs(args):
  """Parses arguments vector, looking for switches of the form -key {optional value}.
  For example:
    parseArgs([ 'main.py', '-e', 20, '-r', 0.1, '-m', 'Simple' ]) = { '-e':20, '-r':5, '-t': 'simple' }"""
  args_map = {}
  curkey = None
  for i in xrange(1, len(args)):
    if args[i][0] == '-':
      args_map[args[i]] = True
      curkey = args[i]
    else:
      assert curkey
      args_map[curkey] = args[i]
      curkey = None
  return args_map

def validateInput(args):
  args_map = parseArgs(args)
  assert '-e' in args_map, "A number of epochs should be specified with the flag -e (ex: -e 10)"
  assert '-r' in args_map, "A learning rate should be specified with the flag -r (ex: -r 0.1)"
  assert '-t' in args_map, "A network type should be provided. Options are: simple | hidden | custom"
  return(args_map)

def main():

  # Parsing command line arguments
  args_map = validateInput(sys.argv)
  epochs = int(args_map['-e'])
  rate = float(args_map['-r'])
  networkType = args_map['-t']
  graph = '-g' in args_map

  if '-n' in args_map:
      number_of_hidden_nodes = int(args_map['-n'])

  if '-l' in args_map:
      number_of_layers = int(args_map['-l'])

  # Load in the training data.
  images = DataReader.GetImages('training-9k.txt', -1)
  for image in images:
    assert len(image.pixels) == 14
    assert len(image.pixels[0]) == 14

  # Load the validation set.
  validation = DataReader.GetImages('validation-1k.txt', -1)
  for image in validation:
    assert len(image.pixels) == 14
    assert len(image.pixels[0]) == 14

  # Initializing network

  if networkType == 'simple':
    network = SimpleNetwork()
  if networkType == 'hidden':
    network = HiddenNetwork(number_of_hidden_nodes)
  if networkType == 'custom':
    network = CustomNetwork(number_of_hidden_nodes, number_of_layers)

  # Hooks user-implemented functions to network
  network.FeedForwardFn = FeedForward

  # If network type is custom, we use decaying function
  if networkType != 'custom':
      network.TrainFn = Train

  # Initialize network weights
  network.InitializeWeights()

  # Displays information
  print '* * * * * * * * *'
  print 'Parameters => Epochs: %d, Learning Rate: %f' % (epochs, rate)
  print 'Type of network used: %s' % network.__class__.__name__
  print ('Input Nodes: %d, Hidden Nodes: %d, Output Nodes: %d' %
         (len(network.network.inputs), len(network.network.hidden_nodes),
          len(network.network.outputs)))
  print '* * * * * * * * *'
  # Train the network.
  epochs, data = network.Train(images, validation, rate, epochs)
  data = data[1::]

  test_images = DataReader.GetImages('test-1k.txt', -1)
  test_performance = network.Performance(test_images)
  print "Performance on test data: {0}".format(test_performance)

  if graph:
      plt.clf()
      plt.title('Performance of Hidden Neural Network with 30 Hidden Units vs. Epoch')
      plt.xlabel('Epoch')
      plt.ylabel('Performance')

      y_training = map(lambda x: x[0], data)
      y_test = map(lambda x: x[1], data)

      lower_bound = max(min(y_test) - .1, 0.)
      plt.axis([0, epochs, lower_bound, 1.0])

      xs = range(1, epochs+1)
      p1, = plt.plot(xs, y_training, color='b')
      p2, = plt.plot(xs, y_test, color='r')

      plt.legend((p1,p2,), ('Training Data', 'Test Data'), 'lower right')
      plt.savefig('hidden30.pdf')

if __name__ == "__main__":
  main()

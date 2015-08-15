import numpy, struct, array, os, pickle

import neural_net

print "Opening training data files..."
imagesFile = open(os.path.dirname(__file__) + "train-images.idx3-ubyte", "r")
imagesMagicNumber, numberImages, rows, columns = struct.unpack(">IIII", imagesFile.read(16))
assert imagesMagicNumber == 2051

labelsFile = open(os.path.dirname(__file__) + "train-labels.idx1-ubyte", "r")
labelsMagicNumber, numberLabels = struct.unpack(">II", labelsFile.read(8))
assert labelsMagicNumber == 2049

assert numberImages == numberLabels

print "Reading training data files..."
data = []
imageData = array.array("B", imagesFile.read())
labelData = array.array("B", labelsFile.read())

print "Preparing training data..."
testCaseInput = None
testCaseOutput = None
for testCaseIndex in range(numberImages):
	testCaseInput = numpy.asarray(imageData[testCaseIndex * rows * columns : (testCaseIndex + 1) * rows * columns], dtype = float)
	testCaseOutput = numpy.zeros(10)
	testCaseOutput[labelData[testCaseIndex]] = 1
	data.append((testCaseInput, testCaseOutput))

print "Initializing neural net..."
neural_net = neural_net.sigmoid_quadratic_backpropogation_neural_net([784, 100, 100, 10])

print "Training..."
epoch_size = 100
epoch_index = 0
while True:
	training_accuracy = 0
	for batch_index in range(epoch_size):
		training_accuracy += neural_net.train(data, 100, 0.001)
	print "Epoch #" + str(epoch_index) + ": " + "{:.1%}".format(training_accuracy / epoch_size) + " accuracy"
	if epoch_index == 29:
		pickle.dump(neural_net, open(os.path.dirname(__file__) + "neural_net", "w"))
	epoch_index += 1
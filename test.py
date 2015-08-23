import numpy, struct, array, os, pickle, random

import neural_net, population

print("Opening training data files...")
imagesFile = open(os.path.dirname(__file__) + "train-images.idx3-ubyte", "r")
imagesMagicNumber, numberImages, rows, columns = struct.unpack(">IIII", imagesFile.read(16))
assert imagesMagicNumber == 2051

labelsFile = open(os.path.dirname(__file__) + "train-labels.idx1-ubyte", "r")
labelsMagicNumber, numberLabels = struct.unpack(">II", labelsFile.read(8))
assert labelsMagicNumber == 2049

assert numberImages == numberLabels

print("Reading training data files...")
data = []
imageData = array.array("B", imagesFile.read())
labelData = array.array("B", labelsFile.read())

print("Preparing training data...")
testCaseInput = None
testCaseOutput = None
for testCaseIndex in range(numberImages):
	testCaseInput = numpy.asarray(imageData[testCaseIndex * rows * columns : (testCaseIndex + 1) * rows * columns], dtype = float)
	testCaseOutput = numpy.zeros(10)
	testCaseOutput[labelData[testCaseIndex]] = 1
	data.append((testCaseInput, testCaseOutput))

print("Initializing sigmoid quadratic backpropogation neural net...")
test_neural_net = neural_net.sigmoid_quadratic_backpropogation_neural_net([784, 100, 100, 10])

print("Training...")
epoch_size = 100
epoch_index = 0
while epoch_index < 0:
	training_accuracy = 0
	for batch_index in range(epoch_size):
		training_accuracy += test_neural_net.train(data, 100, 0.001)
	print("Epoch #" + str(epoch_index) + ": " + "{:.1%}".format(training_accuracy / epoch_size) + " accuracy")
	if epoch_index == 29:
		pickle.dump(test_neural_net, open(os.path.dirname(__file__) + "neural_net", "w"))
	epoch_index += 1

inputs = []
outputs = []
for test_case_index in range(100):
	test_case = data[random.randint(0, len(data) - 1)]
	inputs.append(test_case[0])
	outputs.append(test_case[1])
inputs = numpy.vstack(inputs)
outputs = numpy.vstack(outputs)
data = (inputs, outputs)

print("Initializing neural net population...")
test_population = population.neural_net_population(population.sigmoid_quadratic_backpropogation_neural_net_individual(0.01, 0.0001, [784, 100, 100, 10]), 100, data)
print("Evolving...")
while True:
	print(test_population.select())
	test_population.evolve()
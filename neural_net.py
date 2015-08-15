import numpy, random, math, sys, numpy

class base_neural_net:
	pass

class base_backpropogation_neural_net(base_neural_net):
	def __init__(self, shape):
		numpy.seterr(all = 'raise')
		self._shape = shape
		self._weights = [None for layerIndex in range(len(self._shape))]
		for layerIndex in range(1, len(self._shape)):
			self._weights[layerIndex] = numpy.random.standard_normal([self._shape[layerIndex], self._shape[layerIndex - 1]]) / math.sqrt(self._shape[layerIndex - 1])
		self._weighted_inputs = [None for layerIndex in range(len(self._shape))]
		self._biases = [None for layerIndex in range(len(self._shape))]
		for layerIndex in range(1, len(self._shape)):
			self._biases[layerIndex] = numpy.random.standard_normal([self._shape[layerIndex]]) / math.sqrt(self._shape[layerIndex - 1])
		self._activations = [None for layerIndex in range(len(self._shape))]
		self._errors = [None for layerIndex in range(len(self._shape))]

	def train(self, dataset, batch_size, learning_rate):
		inputs = []
		outputs = []
		for test_case_index in range(batch_size):
			test_case = dataset[random.randint(0, len(dataset) - 1)]
			inputs.append(test_case[0])
			outputs.append(test_case[1])
		inputs = numpy.vstack(inputs)
		outputs = numpy.vstack(outputs)
		self.run(inputs)
		self._learn(outputs, batch_size, learning_rate)
		classifications = numpy.argmax(self._activations[-1], 1)
		answers = numpy.argmax(outputs, 1)
		number_test_cases_correct = 0
		for test_case_index in range(len(classifications)):
			if classifications[test_case_index] == answers[test_case_index]:
				number_test_cases_correct += 1
		return float(number_test_cases_correct) / batch_size

	def run(self, inputs):
		self._activations[0] = inputs
		for layer_index in range(1, len(self._shape)):
			self._weighted_inputs[layer_index] = numpy.clip((self._activations[layer_index - 1][:, numpy.newaxis, :] * self._weights[layer_index][numpy.newaxis, :, :]).sum(2), math.log(sys.float_info.min) * 0.9, math.log(math.sqrt(sys.float_info.max) - 1) * 0.9)
			self._weighted_inputs[layer_index] += self._biases[layer_index][numpy.newaxis, :]
			self._activations[layer_index] = self._activation_function(layer_index)

	def _learn(self, outputs, batch_size, learning_rate):
		for layer_index in reversed(range(1, len(self._shape))):
			if layer_index == len(self._shape) - 1:
				self._errors[layer_index] = self._cost_function_derivative(outputs) * self._activation_function_derivative(layer_index)
			else:
				self._errors[layer_index] = (numpy.transpose(self._weights[layer_index + 1])[numpy.newaxis, :, :] * self._errors[layer_index + 1][:, numpy.newaxis, :]).sum(2) * self._activation_function_derivative(layer_index)
			self._weights[layer_index] -= learning_rate * (self._errors[layer_index][:, :, numpy.newaxis] * self._activations[layer_index - 1][:, numpy.newaxis, :]).sum(0)
			self._biases[layer_index] -= learning_rate * self._errors[layer_index].sum(0)

class sigmoid_quadratic_backpropogation_neural_net(base_backpropogation_neural_net):
	def _activation_function(self, layer_index):
		return numpy.power(1 + numpy.exp(-self._weighted_inputs[layer_index]), -1)

	def _activation_function_derivative(self, layer_index):
		return numpy.exp(self._weighted_inputs[layer_index]) / numpy.power(numpy.exp(self._weighted_inputs[layer_index]) + 1, 2)

	def _cost_function_derivative(self, outputs):
		return self._activations[-1] - outputs
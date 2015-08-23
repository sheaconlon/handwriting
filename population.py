import random, numpy

import neural_net

class base_population(object):
	def __init__(self, first_individual, size):
		self._size = size
		self._individuals = [first_individual]
		self._fitnesses = numpy.empty(self._size)
		for individual_index in range(1, self._size):
			self._individuals.append(first_individual.random())

	def evolve(self):
		new_individuals = []
		for new_individual_index in range(self._size):
			parent_one = self._choose_parent()
			parent_two = self._choose_parent()
			new_individuals.append(parent_one.mate(parent_two))
		self._individuals = new_individuals

	def _choose_parent(self):
		target_cumulative_fitness = random.uniform(0, self._total_fitness)
		individual_index = 0
		cumulative_fitness = self._fitnesses[0]
		while cumulative_fitness < target_cumulative_fitness:
			individual_index += 1
			cumulative_fitness += self._fitnesses[individual_index]
		return self._individuals[individual_index]

class neural_net_population(base_population):
	def __init__(self, first_individual, size, test_data):
		super(neural_net_population, self).__init__(first_individual, size)
		self._test_data = test_data

	def select(self):
		self._total_fitness = 0
		for individual_index in range(self._size):
			self._fitnesses[individual_index] = self._individuals[individual_index].fitness(self._test_data)
			self._total_fitness += self._fitnesses[individual_index]
		best = self._individuals[numpy.argmax(self._fitnesses)]
		classifications = numpy.argmax(best._neural_net._activations[-1], 1)
		answers = numpy.argmax(self._test_data[1], 1)
		number_test_cases_correct = 0
		for test_case_index in range(len(classifications)):
			if classifications[test_case_index] == answers[test_case_index]:
				number_test_cases_correct += 1
		return number_test_cases_correct

class base_individual(object):
	def __init__(self, crossover_probability, mutation_probability, genetic_schema):
		self._crossover_probability = crossover_probability
		self._mutation_probability = mutation_probability
		self._genetic_schema = genetic_schema

	def mate(self, other_parent):
		child = self.empty()
		for chromosome_index in range(len(child._genetic_schema)):
			if random.random() < self._crossover_probability:
				crossover_gene = random.randint(0, child._genetic_schema[chromosome_index] - 1)
				child._chromosomes[chromosome_index] = numpy.copy(self._chromosomes[chromosome_index])
				for gene_index in range(crossover_gene, child._genetic_schema[chromosome_index]):
					child._chromosomes[chromosome_index][gene_index] = other_parent._chromosomes[chromosome_index][gene_index]
			else:
				if random.random() < 0.5:
					child._chromosomes[chromosome_index] = numpy.copy(self._chromosomes[chromosome_index])
				else:
					child._chromosomes[chromosome_index] = numpy.copy(other_parent._chromosomes[chromosome_index])
			for gene_index in range(child._genetic_schema[chromosome_index]):
				if random.random() < self._mutation_probability:
					child._chromosomes[chromosome_index][gene_index] = child._random_gene()
		return child


class base_float_individual(base_individual):
	def _random_gene(self):
		return random.random() * 2 - 1

class sigmoid_quadratic_backpropogation_neural_net_individual(base_float_individual):
	def __init__(self, crossover_probability, mutation_probability, shape):
		self._neural_net = neural_net.sigmoid_quadratic_backpropogation_neural_net(shape)
		genetic_schema = []
		self._chromosomes = []
		for layer_index in range(1, len(shape)):
			genetic_schema.append(shape[layer_index - 1] * shape[layer_index])
			self._chromosomes.append(self._neural_net._weights[layer_index].flatten())
		for layer_index in range(1, len(shape)):
			genetic_schema.append(shape[layer_index])
			self._chromosomes.append(self._neural_net._biases[layer_index])
		super(sigmoid_quadratic_backpropogation_neural_net_individual, self).__init__(crossover_probability, mutation_probability, genetic_schema)

	def random(self):
		return sigmoid_quadratic_backpropogation_neural_net_individual(self._crossover_probability, self._mutation_probability, self._neural_net._shape)

	def empty(self):
		return sigmoid_quadratic_backpropogation_neural_net_individual(self._crossover_probability, self._mutation_probability, self._neural_net._shape)

	def fitness(self, test_data):
		for layer_index in range(1, len(self._neural_net._shape)):
			self._neural_net._weights[layer_index] = self._chromosomes[layer_index - 1].reshape((self._neural_net._shape[layer_index], self._neural_net._shape[layer_index - 1]))
		for layer_index in range(1, len(self._neural_net._shape)):
			self._neural_net._biases[layer_index] = self._chromosomes[len(self._neural_net._shape) - 1 + layer_index - 1].reshape(self._neural_net._shape[layer_index])
		self._neural_net.run(test_data[0])
		return 1 / numpy.sum(numpy.absolute(self._neural_net._activations[-1] - test_data[1]))
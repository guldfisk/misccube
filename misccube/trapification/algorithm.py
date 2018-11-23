import typing as t

import math
import random
import itertools
import statistics

import numpy as np
import matplotlib.pyplot as plt

from deap import base, creator, tools, algorithms

from mtgorp.models.persistent.attributes.colors import Color
from mtgorp.utilities.containers import HashableMultiset

from magiccube.laps.traps.tree.printingtree import AllNode, PrintingNode
from magiccube.laps.traps.trap import Trap


class ConstrainedNode(object):

	def __init__(self, value: float, node: PrintingNode, groups: t.Iterable[str] = ()):
		self.value = value
		self.node = node

		self.groups = frozenset(
			itertools.chain(
				groups,
				(
					color.name
					for color in
					Color
					if (
						len(node.children) == 1
						and node.children.__iter__().__next__().cardboard.front_card.color == frozenset((color,))
					)
				)
			),
		)

		self._hash = hash(self.node)

	def __repr__(self):
		return f'CC({self.value})'

	def __deepcopy__(self, memodict: t.Dict):
		return self


class TrapDistribution(object):

	fitness = None #type: base.Fitness

	def __init__(
		self,
		constrained_nodes: t.Iterable[ConstrainedNode] = (),
		trap_amount: int = 1,
		traps: t.Optional[t.List[t.List[ConstrainedNode]]] = None,
		random_initialization: bool = False
	):
		if traps is None:
			self.traps = [[] for _ in range(trap_amount)] #type: t.List[t.List[ConstrainedNode]]

			if random_initialization:
				for constrained_node in constrained_nodes:
					random.choice(self.traps).append(constrained_node)

			else:
				for constrained_node, i in zip(constrained_nodes, itertools.cycle(range(trap_amount))):
					self.traps[i].append(constrained_node)

		else:
			self.traps = traps #type: t.List[t.List[ConstrainedNode]]

	@property
	def as_trap_collection(self) -> HashableMultiset[Trap]:
		traps = []

		for trap in self.traps:
			cubeables = []

			if not trap:
				raise Exception('Empty trap')

			for constrained_node in trap:
				cubeable = constrained_node.node
				if isinstance(cubeable, AllNode):
					cubeables.extend(cubeable.children)

				else:
					cubeables.append(cubeable)

			traps.append(Trap(AllNode(cubeables)))

		return HashableMultiset(traps)

	def __repr__(self) -> str:
		return f'{self.__class__.__name__}({self.traps})'


def mutate_cubeables_groups(cubeables_groups: TrapDistribution) -> t.Tuple[TrapDistribution]:
	for i in range(3):
		selected_group = random.choice(
			[
				group
				for group in
				cubeables_groups.traps
				if group
			]
		)
		target_group = random.choice(
			[
				group
				for group in
				cubeables_groups.traps
				if group != selected_group
			]
		)
		target_group.append(
			selected_group.pop(
				random.randint(
					0,
					len(selected_group) - 1,
				)
			)
		)
	return cubeables_groups,


def mate(
	distribution_1: TrapDistribution,
	distribution_2: TrapDistribution,
	distributor: 'Distributor',
) -> t.Tuple[TrapDistribution, TrapDistribution]:

	locations = {
		node: []
		for node in
		distributor.constrained_nodes
	}
	for grouping in (distribution_1, distribution_2):
		for i in range(len(grouping.traps)):
			for node in grouping.traps[i]:
				locations[node].append(i)

	for grouping in (distribution_1, distribution_2):
		groups = [[] for _ in range(distributor.trap_amount)]

		for node, possibilities in locations.items():
			groups[random.choice(possibilities)].append(node)

		grouping.traps = groups

	return distribution_1, distribution_2


def logistic(x: float, max_value: float, mid: float, slope: float) -> float:
	return max_value / (1 + math.e ** (slope * (x - mid)))


def value_distribution_homogeneity_factor(distribution: TrapDistribution, distributor: 'Distributor') -> float:
	return sum(
		(
			abs(
				sum(
					cubeable.value
					for cubeable in
					trap
				) - distributor.expected_trap_value
			) ** 2
			for trap in
			distribution.traps
		)
	) / distributor.max_trap_value_deviation


def value_distribution_homogeneity_score(distribution: TrapDistribution, distributor: 'Distributor') -> float:
	return logistic(
		x = value_distribution_homogeneity_factor(
			distribution,
			distributor
		),
		max_value = 2,
		mid = 0,
		slope = 100,
	)


def _group_collision_factor(distribution: TrapDistribution, distributor: 'Distributor') -> int:
	collision_factor = 0

	for trap in distribution.traps:
		groups = {} #type: t.Dict[str, t.List[ConstrainedNode]]
		collisions = {} #type: t.Dict[t.FrozenSet[ConstrainedNode], t.List[str]]

		for constrained_node in trap:
			for group in constrained_node.groups:

				if group in groups:
					for other_node in groups[group]:
						_collision_key = frozenset((constrained_node, other_node))
						try:
							collisions[_collision_key].append(group)
						except KeyError:
							collisions[_collision_key] = [group]

					groups[group].append(constrained_node)

				else:
					groups[group] = [constrained_node]

		collision_factor += sum(
			(
				sum(node.value for node in nodes)
				* max(distributor.get_group_weight(group) for group in groups)
			) ** 1.5
			for nodes, groups in
			collisions.items()
		)

	return collision_factor


def group_exclusivity_factor(distribution: TrapDistribution, distributor: 'Distributor') -> float:
	return _group_collision_factor(
		distribution,
		distributor
	) / (distributor.average_group_collision_factor + 1)


def group_exclusivity_score(distribution: TrapDistribution, distributor: 'Distributor') -> float:
	return logistic(
		x = group_exclusivity_factor(
			distribution,
			distributor,
		),
		max_value = 2,
		mid = 0,
		slope = 15,
	)


def trap_size_homogeneity_factor(distribution: TrapDistribution, distributor: 'Distributor') -> float:
	return sum(
		abs(
			len(trap) - distributor.average_trap_size
		) ** 2
		for trap in
		distribution.traps
	) / distributor.max_trap_size_heterogeneity


def size_homogeneity_score(distribution: TrapDistribution, distributor: 'Distributor') -> float:
	return logistic(
		x = trap_size_homogeneity_factor(
			distribution,
			distributor,
		),
		max_value = 2,
		mid = 0,
		slope = 100,
	)


def trap_distribution_score(distribution: TrapDistribution, distributor: 'Distributor') -> t.Tuple[float]:
	return (
			   value_distribution_homogeneity_score(
			distribution,
			distributor,
		) ** 2
			   * group_exclusivity_score(
			distribution,
			distributor,
		) ** 2
			   * size_homogeneity_score(
			distribution,
			distributor,
		)
   ),


class Distributor(object):

	def __init__(
		self,
		constrained_nodes: t.Iterable[ConstrainedNode],
		trap_amount: int,
		group_weights: t.Dict[str, float] = None,
		mate_chance: float = .5,
		mutate_chance: float = .2,
		tournament_size: int = 3,
	):
		self._constrained_nodes = list(constrained_nodes)
		self._trap_amount = trap_amount
		self._group_weights = {} if group_weights is None else group_weights
		self._mate_chance = mate_chance
		self._mutate_chance = mutate_chance
		self._tournament_size = tournament_size

		self._average_trap_size = len(self._constrained_nodes) / self._trap_amount
		self._max_trap_size_heterogeneity = (
			(len(self._constrained_nodes) - self._average_trap_size) ** 2
			+ self._average_trap_size ** 2 * (self._trap_amount - 1)
		)

		_sum_values = sum(
			(
				cubeable.value
				for cubeable in
				self._constrained_nodes
			)
		)

		self._average_trap_value = _sum_values / self._trap_amount
		self._max_trap_value_deviation = (
			 (_sum_values - self._average_trap_value) ** 2
			 + self._average_trap_value ** 2 * (self._trap_amount - 1)
		)

		self._expected_trap_size = len(self._constrained_nodes) / self._trap_amount

		self._average_group_collision_factor = None #type: float

		creator.create('Fitness', base.Fitness, weights=(1.,))
		creator.create('Individual', TrapDistribution, fitness=creator.Fitness)

		self._toolbox = base.Toolbox()

		self._toolbox.register(
			'individual',
			creator.Individual,
			constrained_nodes = self._constrained_nodes,
			trap_amount = self._trap_amount,
			random_initialization = True,
		)
		self._toolbox.register('population', tools.initRepeat, list, self._toolbox.individual)

		self._toolbox.register('evaluate', trap_distribution_score, distributor=self)
		self._toolbox.register('mate', mate, distributor=self)
		self._toolbox.register('mutate', mutate_cubeables_groups)
		self._toolbox.register('select', tools.selTournament, tournsize=self._tournament_size)

		averaging_group_amount = 100

		self._average_group_collision_factor = statistics.mean(
			_group_collision_factor(
				TrapDistribution(
					constrained_nodes=self._constrained_nodes,
					trap_amount=self._trap_amount,
					random_initialization=True,
				),
				self,
			) for _ in
			range(averaging_group_amount)
		) #type: float

		self._population = self._toolbox.population(n=300) #type: t.List[TrapDistribution]
		self._best = None

		self._statistics = tools.Statistics(key=lambda ind: ind)
		self._statistics.register('avg', lambda s: statistics.mean(e.fitness.values[0] for e in s))
		self._statistics.register('max', lambda s: max(e.fitness.values[0] for e in s))
		self._statistics.register('val', lambda s: max(value_distribution_homogeneity_score(e, self) for e in s))
		self._statistics.register('siz', lambda s: max(size_homogeneity_score(e, self) for e in s))
		self._statistics.register('grp', lambda s: max(group_exclusivity_score(e, self) for e in s))

		self._logbook = None #type: tools.Logbook

	@property
	def population(self) -> t.List[TrapDistribution]:
		return self._population

	@property
	def average_trap_size(self) -> float:
		return self._average_trap_size

	@property
	def max_trap_size_heterogeneity(self) -> float:
		return self._max_trap_size_heterogeneity

	@property
	def best(self) -> TrapDistribution:
		if self._best is None:
			self._best = tools.selBest(self._population, 1)[0]

		return self._best

	@property
	def constrained_nodes(self) -> t.List[ConstrainedNode]:
		return self._constrained_nodes

	@property
	def trap_amount(self) -> int:
		return self._trap_amount

	@property
	def expected_trap_value(self) -> float:
		return self._average_trap_value

	@property
	def max_trap_value_deviation(self) -> float:
		return self._max_trap_value_deviation

	@property
	def expected_trap_size(self) -> float:
		return self._expected_trap_size

	@property
	def average_group_collision_factor(self) -> float:
		return self._average_group_collision_factor

	def get_group_weight(self, group: str) -> float:
		return self._group_weights.get(group, 1.)

	def evaluate_cube(self, traps: t.Collection[Trap]) -> float:

		index_map = {}  # type: t.Dict[PrintingNode, int]

		for index, trap in enumerate(traps):
			for child in trap.node.children:
				index_map[child if isinstance(child, PrintingNode) else AllNode((child,))] = index

		traps = [[] for _ in range(len(traps))]

		for node in self._constrained_nodes:
			try:
				traps[index_map[node.node]].append(node)
			except KeyError as e:
				if isinstance(node.node, AllNode):
					traps[index_map[AllNode((node.node.children.__iter__().__next__(),))]].append(node)
				else:
					raise e

		distribution = TrapDistribution(traps=traps)

		return trap_distribution_score(
			distribution,
			self,
		)[0]

	def show_plot(self) -> 'Distributor':
		generations = self._logbook.select("gen")
		fit_maxes = self._logbook.select('max')
		fit_averages = self._logbook.select('avg')

		value_distribution_homogeneity_scores = self._logbook.select('val')
		size_homogeneity_scores = self._logbook.select('siz')
		group_exclusivity_scores = self._logbook.select('grp')

		fig, ax1 = plt.subplots()

		max_line = plt.plot(generations, fit_maxes, 'k', label='Maximum Fitness')
		average_line = plt.plot(generations, fit_averages, '.75', label='Average Fitness')
		value_line = plt.plot(generations, value_distribution_homogeneity_scores, 'r', label='Max Value Distribution Homogeneity Score')
		size_line = plt.plot(generations, size_homogeneity_scores, 'g', label='Max Size Homogeneity Score')
		group_line = plt.plot(generations, group_exclusivity_scores, 'b', label='Max Group Exclusivity Score')

		lns = max_line + average_line + value_line + size_line + group_line
		labs = [l.get_label() for l in lns]
		ax1.legend(lns, labs, loc="bottom right")

		plt.show()

		return self

	def evaluate(self, generations: int) -> 'Distributor':
		population, logbook = algorithms.eaSimple(
			self._population,
			self._toolbox,
			self._mate_chance,
			self._mutate_chance,
			generations,
			stats = self._statistics,
		)

		self._population = population
		self._logbook = logbook
		self._best = None

		return self

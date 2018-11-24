import typing as t

import math
import random
import itertools
import statistics
import functools
import operator

from abc import ABC, abstractmethod

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


creator.create('Fitness', base.Fitness, weights=(1.,))
creator.create('Individual', TrapDistribution, fitness=creator.Fitness)


def mutate_trap_distribution(distribution: TrapDistribution) -> t.Tuple[TrapDistribution]:
	for i in range(3):
		selected_group = random.choice(
			[
				group
				for group in
				distribution.traps
				if group
			]
		)
		target_group = random.choice(
			[
				group
				for group in
				distribution.traps
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
	return distribution,


def mate_distributions(
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


class Constraint(ABC):
	description = 'A constraint' #type: str

	def __init__(
		self,
		constrained_nodes: t.FrozenSet[ConstrainedNode],
		trap_amount: int,
		random_population: t.Collection[TrapDistribution],
	):
		self._constrained_nodes = constrained_nodes
		self._trap_amount = trap_amount

	@abstractmethod
	def score(self, distribution: TrapDistribution) -> float:
		pass


class ConstraintSet(object):

	def __init__(self, constraints: t.Tuple[t.Tuple[Constraint, float], ...]):
		self._constraints = constraints

	def score(self, distribution: TrapDistribution) -> float:
		return functools.reduce(
			operator.mul,
			(
				constraint.score(distribution) ** weight
				for constraint, weight in
				self._constraints
			)
		)

	def __iter__(self) -> t.Iterable[Constraint]:
		return (constraint for constraint, _ in self._constraints)
	
	
class ConstraintSetBluePrint(object):

	def __init__(self, *constraints: t.Tuple[t.Type[Constraint], float, t.Dict]):
		self._constraints = constraints
		
	def realise(
		self,
		constrained_nodes: t.FrozenSet[ConstrainedNode],
		trap_amount: int,
		random_population: t.Collection[TrapDistribution],
	) -> ConstraintSet:
		return ConstraintSet(
			tuple(
				(
					constraint_type(constrained_nodes, trap_amount, random_population, **kwargs),
					weight,
				)
				for constraint_type, weight, kwargs in
				self._constraints
			)
		)


class ValueDistributionHomogeneityConstraint(Constraint):
	description = 'Value distribution homogeneity'

	def __init__(
		self,
		nodes: t.FrozenSet[ConstrainedNode],
		trap_amount: int,
		random_population: t.Collection[TrapDistribution],
	):
		super().__init__(nodes, trap_amount, random_population)
		self._average_trap_value = (
			sum(
				(
					cubeable.value
					for cubeable in
					self._constrained_nodes
				)
			) / self._trap_amount
		)

		self._average_value_distribution_homogeneity_factor = statistics.mean(
			self._value_distribution_heterogeneity_factor(
				distribution
			)
			for distribution in
			random_population
		)

	def _value_distribution_heterogeneity_factor(self, distribution: TrapDistribution) -> float:
		return sum(
			(
				abs(
					sum(
						cubeable.value
						for cubeable in
						trap
					) - self._average_trap_value
				) ** 2
				for trap in
				distribution.traps
			)
		)

	def _value_distribution_homogeneity_factor(self, distribution: TrapDistribution) -> float:
		return self._value_distribution_heterogeneity_factor(
			distribution
		) / self._average_value_distribution_homogeneity_factor

	def score(self, distribution: TrapDistribution) -> float:
		return logistic(
			x = self._value_distribution_homogeneity_factor(
				distribution
			),
			max_value = 2,
			mid = 0,
			slope = 15,
		)


class DivergenceConstraint(Constraint):
	description = 'Divergence'

	def __init__(
		self,
		constrained_nodes: t.FrozenSet[ConstrainedNode],
		trap_amount: int,
		random_population: t.Collection[TrapDistribution],
		origin: HashableMultiset[Trap],
	):
		super().__init__(constrained_nodes, trap_amount, random_population)
		self._origin = origin

	def score(self, distribution: TrapDistribution) -> float:
		return logistic(
			x = len(self._origin - distribution.as_trap_collection) / len(self._origin),
			max_value = 2,
			mid = 0,
			slope = 15,
		)


class GroupExclusivityConstraint(Constraint):
	description = 'Group exclusivity'

	def __init__(
		self,
		nodes: t.FrozenSet[ConstrainedNode],
		trap_amount: int,
		random_population: t.Collection[TrapDistribution],
		group_weights: t.Dict[str, float],
	):
		super().__init__(nodes, trap_amount, random_population)
		self._group_weights = {} if group_weights is None else group_weights

		self._average_group_collision_factor = statistics.mean(
			self._group_collision_factor(
				distribution
			)
			for distribution in
			random_population
		)
		
	def _get_group_weight(self, group: str) -> float:
		return self._group_weights.get(group, 1.)

	def _group_collision_factor(self, distribution: TrapDistribution) -> int:
		collision_factor = 0

		for trap in distribution.traps:
			groups = {}  # type: t.Dict[str, t.List[ConstrainedNode]]
			collisions = {}  # type: t.Dict[t.FrozenSet[ConstrainedNode], t.List[str]]

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
					* max(self._get_group_weight(group) for group in groups)
				) ** 1.5
				for nodes, groups in
				collisions.items()
			)

		return collision_factor

	def _group_exclusivity_factor(self, distribution: TrapDistribution) -> float:
		return self._group_collision_factor(
			distribution
		) / (self._average_group_collision_factor + 1)

	def score(self, distribution: TrapDistribution) -> float:
		return logistic(
			x = self._group_exclusivity_factor(
				distribution
			),
			max_value = 2,
			mid = 0,
			slope = 15,
		)


class SizeHomogeneityConstraint(Constraint):
	description = 'Size homogeneity'

	def __init__(
		self,
		constrained_nodes: t.FrozenSet[ConstrainedNode],
		trap_amount: int,
		random_population: t.Collection[TrapDistribution],
	):
		super().__init__(constrained_nodes, trap_amount, random_population)
		self._average_trap_size = len(self._constrained_nodes) / self._trap_amount

		self._average_size_homogeneity_factor = statistics.mean(
			self._size_heterogeneity_factor(
				distribution
			)
			for distribution in
			random_population
		)

	def _size_heterogeneity_factor(self, distribution: TrapDistribution) -> float:
		return sum(
			abs(
				len(trap) - self._average_trap_size
			) ** 2
			for trap in
			distribution.traps
		)

	def _size_homogeneity_factor(self, distribution: TrapDistribution) -> float:
		return self._size_heterogeneity_factor(
			distribution
		) / self._average_size_homogeneity_factor

	def score(self, distribution: TrapDistribution) -> float:
		return logistic(
			x = self._size_homogeneity_factor(
				distribution
			),
			max_value = 2,
			mid = 0,
			slope = 15,
		)


class Distributor(object):

	def __init__(
		self,
		constrained_nodes: t.Iterable[ConstrainedNode],
		trap_amount: int,
		constraint_set_blue_print: ConstraintSetBluePrint,
		# group_weights: t.Dict[str, float] = None,
		mate_chance: float = .5,
		mutate_chance: float = .2,
		tournament_size: int = 3,
		population_size: int = 300,
		seed_trap_collection: t.Optional[t.Collection[Trap]] = None,
	):
		self._constrained_nodes = frozenset(constrained_nodes)
		self._trap_amount = trap_amount
		self._constraint_set_blue_print = constraint_set_blue_print
		self._mate_chance = mate_chance
		self._mutate_chance = mutate_chance
		self._tournament_size = tournament_size
		self._population_size = population_size

		self._seed_trap_collection = seed_trap_collection


		self._toolbox = base.Toolbox()

		self._toolbox.register(
			'individual',
			creator.Individual,
			constrained_nodes = self._constrained_nodes,
			trap_amount = self._trap_amount,
			random_initialization = True,
		)
		self._toolbox.register('population', tools.initRepeat, list, self._toolbox.individual)

		self._population = self._toolbox.population(n=self._population_size) #type: t.List[TrapDistribution]
		self._sample_random_population = self._population

		self._constraint_set = self._constraint_set_blue_print.realise(
			self._constrained_nodes,
			self._trap_amount,
			self._sample_random_population,
		)

		if self._seed_trap_collection is not None:
			self._population = [
				self._toolbox.individual(
					traps = self._trap_collection_to_trap_distribution(
						self._seed_trap_collection
					).traps
				)
				for _ in
				range(self._population_size)
			]

		self._toolbox.register('evaluate', lambda d: (self._constraint_set.score(d),))
		self._toolbox.register('mate', mate_distributions, distributor=self)
		self._toolbox.register('mutate', mutate_trap_distribution)
		self._toolbox.register('select', tools.selTournament, tournsize=self._tournament_size)

		self._best = None

		self._statistics = tools.Statistics(key=lambda ind: ind)
		self._statistics.register('avg', lambda s: statistics.mean(e.fitness.values[0] for e in s))
		self._statistics.register('max', lambda s: max(e.fitness.values[0] for e in s))

		class _MaxMap(object):

			def __init__(self, _constraint: Constraint):
				self._constraint = _constraint

			def __call__(self, population: t.Collection[TrapDistribution]):
				return max(map(self._constraint.score, population))

		for constraint in self._constraint_set:
			self._statistics.register(
				constraint.description,
				_MaxMap(constraint)
			)

		self._logbook = None #type: tools.Logbook

	@property
	def population(self) -> t.List[TrapDistribution]:
		return self._population

	@property
	def sample_random_population(self) -> t.List[TrapDistribution]:
		return self._sample_random_population

	@property
	def best(self) -> TrapDistribution:
		if self._best is None:
			self._best = tools.selBest(self._population, 1)[0]

		return self._best

	@property
	def constrained_nodes(self) -> t.FrozenSet[ConstrainedNode]:
		return self._constrained_nodes

	@property
	def trap_amount(self) -> int:
		return self._trap_amount

	@property
	def constraint_set(self) -> ConstraintSet:
		return self._constraint_set

	def _trap_collection_to_trap_distribution(self, traps: t.Collection[Trap]) -> TrapDistribution:
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

		return TrapDistribution(traps=traps)

	def evaluate_cube(self, traps: t.Collection[Trap]) -> float:
		return self._constraint_set.score(
			self._trap_collection_to_trap_distribution(traps)
		)

	def show_plot(self) -> 'Distributor':
		generations = self._logbook.select("gen")
		fit_maxes = self._logbook.select('max')
		fit_averages = self._logbook.select('avg')

		fig, ax1 = plt.subplots()

		colors = ('r', 'g', 'b', 'c', 'm', 'y')

		lines = functools.reduce(
			operator.add,
			(
				ax1.plot(
					generations,
					self._logbook.select(constraint.description),
					color,
					label = f'Max {constraint.description} score',
				)
				for constraint, color in
				zip(self._constraint_set, colors)
			),
		)

		ax2 = ax1.twinx()

		max_line = ax2.plot(generations, fit_maxes, 'k', label='Maximum Fitness')
		average_line = ax2.plot(generations, fit_averages, '.75', label='Average Fitness')

		lns = max_line + average_line + lines
		labs = [l.get_label() for l in lns]
		ax1.legend(lns, labs, loc="lower right")

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

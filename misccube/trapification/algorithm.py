import typing as t

import math
import random
import itertools
import statistics

from multiset import Multiset

from deap import base, creator, tools, algorithms

from mtgorp.models.persistent.attributes.colors import Color
from mtgorp.models.persistent.printing import Printing
from mtgorp.utilities.containers import HashableMultiset

from magiccube.collections.cube import cubeable as cubeable_type
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

	def __eq__(self, other):
		return (
			isinstance(other, self.__class__)
			and self.node == other.node
		)

	def __hash__(self):
		return self._hash

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
		self.traps = None #type: t.List[t.List[ConstrainedNode]]

		if traps is None:
			self.traps = [[] for _ in range(trap_amount)]

			if random_initialization:
				for constrained_node in constrained_nodes:
					random.choice(self.traps).append(constrained_node)

			else:
				for constrained_node, i in zip(constrained_nodes, itertools.cycle(range(trap_amount))):
					self.traps[i].append(constrained_node)

		else:
			self.traps = traps

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

	locations = {cubeable: [] for cubeable in distributor.constrained_nodes}
	for grouping in (distribution_1, distribution_2):
		for i in range(len(grouping.traps)):
			for cubeable in grouping.traps[i]:
				locations[cubeable].append(i)

	for grouping in (distribution_1, distribution_2):
		groups = [[] for _ in range(distributor.trap_amount)]
		for cubeable, possibilities in locations.items():
			groups[random.choice(possibilities)].append(cubeable)
		grouping.traps = groups

	return distribution_1, distribution_2


def logistic(x: float, mid: float, slope: float) -> float:
	return 1 / (1 + math.e ** (slope * (x - mid)))


def _value_distribution_homogeneity_score(distribution: TrapDistribution, distributor: 'Distributor') -> float:
	return 1 - sum(
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
	) ** .5 / distributor.max_trap_value_deviation


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
				max(node.value for node in nodes)
				* max(distributor.get_group_weight(group) for group in groups)
			)
			for nodes, groups in
			collisions.items()
		)

	return collision_factor


def _group_exclusivity_score(distribution: TrapDistribution, distributor: 'Distributor') -> float:
	return logistic(
		x =_group_collision_factor(distribution, distributor) / (distributor.average_group_collision_factor + 1),
		mid = 1.,
		slope = 3.,
	)


def _trap_size_homogeneity_score(distribution: TrapDistribution, distributor: 'Distributor') -> float:
	return 1 - sum(
		abs(
			len(trap) - distributor.average_trap_size
		) ** 2
		for trap in
		distribution.traps
	) ** .5 / distributor.max_trap_size_heterogeneity


def distribution_score(distribution: TrapDistribution, distributor: 'Distributor') -> t.Tuple[float]:
	return (
		2 * _value_distribution_homogeneity_score(
			distribution,
			distributor,
		) +
		2 * _group_exclusivity_score(
			distribution,
			distributor,
		) +
		1 * _trap_size_homogeneity_score(
		   distribution,
		   distributor,
		)
   ) / 5,


class Distributor(object):

	def __init__(
		self,
		cubeables: t.Iterable[ConstrainedNode],
		trap_amount: int,
		group_weights: t.Dict[str, float] = None,
	):
		self._constrained_nodes = frozenset(cubeables)
		self._trap_amount = trap_amount
		self._group_weights = {} if group_weights is None else group_weights

		self._average_trap_size = len(self._constrained_nodes) / self._trap_amount
		self._max_trap_size_heterogeneity = (
			(len(self._constrained_nodes) - self._average_trap_size) ** 2
			+ self._average_trap_size ** 2 * (self._trap_amount - 1)
		) ** .5

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
		) ** .5

		self._expected_trap_size = len(self._constrained_nodes) / self._trap_amount

		self._average_group_collision_factor = None #type: float

		self._population = None #type: t.List[TrapDistribution]
		self._best = None #type: TrapDistribution

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
	def constrained_nodes(self) -> t.FrozenSet[ConstrainedNode]:
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

	def evaluate(self, generations: int = 100) -> 'Distributor':
		creator.create('Fitness', base.Fitness, weights=(1.,))
		creator.create('Individual', TrapDistribution, fitness=creator.Fitness)

		toolbox = base.Toolbox()

		toolbox.register(
			'individual',
			creator.Individual,
			constrained_nodes = self._constrained_nodes,
			trap_amount = self._trap_amount,
		)
		toolbox.register('population', tools.initRepeat, list, toolbox.individual)

		toolbox.register('evaluate', distribution_score, distributor=self)
		toolbox.register('mate', mate, distributor=self)
		toolbox.register('mutate', mutate_cubeables_groups)
		toolbox.register('select', tools.selTournament, tournsize=3)

		averaging_group_amount = 100

		self._average_group_collision_factor = statistics.mean(
			_group_collision_factor(
				TrapDistribution(
					constrained_nodes = self._constrained_nodes,
					trap_amount = self._trap_amount,
					random_initialization = True,
				),
				self,
			) for _ in
			range(averaging_group_amount)
		)

		population = toolbox.population(n=300)

		for individual in population:
			individual.fitness.value = toolbox.evaluate(individual)

		print(
			tools.selBest(population, 1)[0].fitness.value
		)

		cxpb, mutpb = .5, .2

		population, stats = algorithms.eaSimple(
			population,
			toolbox,
			cxpb,
			mutpb,
			generations,
		)

		self._population = population
		self._best = None

		return self







if __name__ == '__main__':
	main()
	# test()
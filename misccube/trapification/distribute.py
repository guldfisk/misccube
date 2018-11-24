import typing as t

import time
import random
import os
import statistics

from promise import Promise

from proxypdf.write import save_proxy_pdf

from mtgorp.db.load import Loader
from mtgorp.db.database import CardDatabase
from mtgorp.utilities.containers import HashableMultiset

from mtgimg.load import Loader as ImageLoader

from magiccube.laps.lap import Lap
from magiccube.laps.traps.trap import IntentionType
from magiccube.laps.traps.tree.printingtree import AllNode, PrintingNode
from magiccube.collections.cube import Cube

from misccube.cubeload.load import CubeLoader
from misccube.trapification.fetch import ConstrainedNodeFetcher, ConstrainedCubeablesFetchException
from misccube.trapification.algorithm import ConstrainedNode, Distributor, TrapDistribution, ConstraintSetBluePrint
from misccube.trapification import algorithm
from misccube import paths


GARBAGE_OUT_PATH = os.path.join(paths.OUT_DIR, 'garbage_distribution.pdf')
GARBAGE_NEW_OUT_PATH = os.path.join(paths.OUT_DIR, 'garbage_distribution_new.pdf')
GARBAGE_REMOVED_OUT_PATH = os.path.join(paths.OUT_DIR, 'garbage_distribution_removed.pdf')
GARBAGE_LANDS_OUT_PATH = os.path.join(paths.OUT_DIR, 'garbage_lands_distribution.pdf')


def proxy_laps(
	laps: t.Iterable[Lap],
	image_loader: ImageLoader,
	file_name: str,
	margin_size: float = .1,
	card_margin_size: float = .01,
) -> None:
	promises = tuple(
		image_loader.get_image(lap, save=False)
		for lap in
		laps
	)

	images = Promise.all(
		promises
	).get()

	save_proxy_pdf(
		file = file_name,
		images = images,
		margin_size = margin_size,
		card_margin_size = card_margin_size,
	)


_GROUP_WEIGHTS = {
	'WHITE': 1,
	'BLUE': 1.5,
	'BLACK': 1,
	'RED': 1,
	'GREEN': 1,
	'drawgo': 2,
	'mud': 3,
	'post': 4,
	'midrange': 2,
	'mill': 4,
	'reanimate': 4,
	'burn': 4,
	'hatebear': 2,
	'removal': 1,
	'lock': 3,
	'yardvalue': 3,
	'ld': 3,
	'storm': 4,
	'tezz': 3,
	'lands': 3,
	'shatter': 3,
	'bounce': 3,
	'shadow': 4,
	'stifle': 4,
	'beat': 1,
	'cheat': 4,
	'pox': 3,
	'counter': 3,
	'discard': 2,
	'cantrip': 4,
	'balance': 3,
	'stasis': 4,
	'standstill': 3,
	'whitehate': 4,
	'bluehate': 4,
	'blackhate': 4,
	'redhate': 4,
	'greenhate': 4,
	'antiwaste': 4,
	'delirium': 3,
	'sacvalue': 1,
	'lowtoughnesshate': 4,
	'yardhate': 4,
	'armageddon': 4,
	'stax': 3,
	'bloom': 3,
	'weldingjar': 3,
	'drawhate': 4,
	'pluscard': 3,
	# lands
	'fixing': 3,
	'colorlessvalue': 1,
	'fetchable': 2,
	'indestructable': 4,
	'legendarymatters': 1,
	'sol': 3,
	'manland': 4,
	'storage': 3,
}


GROUP_WEIGHT_EXPONENT = 1.5


GROUP_WEIGHTS = {key: value ** GROUP_WEIGHT_EXPONENT for key, value in _GROUP_WEIGHTS.items()}


def calculate(lands: bool = False):
	random.seed()

	db = Loader.load()
	image_loader = ImageLoader()
	fetcher = ConstrainedNodeFetcher(db)
	cube_loader = CubeLoader(db)

	constrained_nodes = fetcher.fetch_garbage_lands() if lands else fetcher.fetch_garbage()

	print(f'loaded {len(constrained_nodes)} nodes')

	cube = cube_loader.load()

	cube_traps = HashableMultiset(
		trap
		for trap in
		cube.traps
		if trap.intention_type == (
			IntentionType.LAND_GARBAGE
			if lands else
			IntentionType.GARBAGE
		)
	)

	blue_print = ConstraintSetBluePrint(
		(
			algorithm.ValueDistributionHomogeneityConstraint,
			2,
			{},
		),
		(
			algorithm.GroupExclusivityConstraint,
			2,
			{'group_weights': GROUP_WEIGHTS},
		),
		(
			algorithm.SizeHomogeneityConstraint,
			1,
			{},
		),
		# (
		# 	algorithm.DivergenceConstraint,
		# 	4,
		# 	{'origin': cube_traps},
		# )
	)

	distributor = Distributor(
		constrained_nodes = constrained_nodes,
		trap_amount = 22 if lands else 44,
		constraint_set_blue_print = blue_print,
		mate_chance = .35,
		mutate_chance = .15,
		tournament_size = 4,
		# seed_trap_collection = cube_traps,
	)

	cube_fitness = distributor.evaluate_cube(cube_traps)

	random_fitness = statistics.mean(
		map(distributor.constraint_set.score, distributor.population)
	)

	st = time.time()

	winner = distributor.evaluate(100).best

	print(f'Done in {time.time() - st} seconds')

	print('Random fitness:', random_fitness)
	print('Current cube fitness:', cube_fitness)

	print('Winner fitness:', winner.fitness.values[0])

	distributor.show_plot()

	winner_traps = winner.as_trap_collection

	new_traps = winner_traps - cube_traps
	removed_traps = cube_traps - winner_traps

	print('New traps', len(new_traps))

	proxy_laps(
		laps = winner_traps,
		image_loader = image_loader,
		file_name = GARBAGE_LANDS_OUT_PATH if lands else GARBAGE_OUT_PATH,
	)

	proxy_laps(
		laps = new_traps,
		image_loader = image_loader,
		file_name = GARBAGE_LANDS_OUT_PATH if lands else GARBAGE_NEW_OUT_PATH,
	)

	proxy_laps(
		laps = removed_traps,
		image_loader = image_loader,
		file_name = GARBAGE_LANDS_OUT_PATH if lands else GARBAGE_REMOVED_OUT_PATH,
	)

	print('proxying done')


def main():
	lands = False
	calculate(lands)


if __name__ == '__main__':
	main()
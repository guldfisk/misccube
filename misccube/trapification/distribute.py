import typing as t

import time
import random
import os

from promise import Promise

from proxypdf.write import save_proxy_pdf

from mtgorp.db.load import Loader

from mtgimg.load import Loader as ImageLoader

from magiccube.laps.lap import Lap

from misccube.trapification.fetch import ConstrainedCubeableFetcher, ConstrainedCubeablesFetchException
from misccube.trapification.algorithm import ConstrainedNode, Distributor, TrapDistribution
from misccube.trapification import algorithm
from misccube import paths


OUT_PATH = os.path.join(paths.OUT_DIR, 'trap_distribution_out.pdf')


def proxy_laps(
	laps: t.Iterable[Lap],
	image_loader: ImageLoader,
	file_name: str = OUT_PATH,
	margin_size: float = .1,
	card_margin_size: float = .01,
) -> None:
	promises = tuple(
		image_loader.get_image(lap)
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


GROUP_WEIGHTS = {
	'WHITE': 1,
	'BLUE': 1,
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
}

def main():
	random.seed()

	db = Loader.load()

	image_loader = ImageLoader()

	fetcher = ConstrainedCubeableFetcher(db)

	constrained_cubeables = fetcher.fetch()

	# constrained_cubeables = random.sample(constrained_cubeables, len(constrained_cubeables))

	distributor = Distributor(constrained_cubeables, 44, GROUP_WEIGHTS)

	st = time.time()

	winner = distributor.evaluate(130).best

	print(time.time() - st, winner.fitness)
	print(distributor.max_trap_value_deviation, winner.fitness)

	for trap in winner.traps:
		print(trap, sum(constrained_cubeables.value for constrained_cubeables in trap))

	print('--')

	traps = winner.as_trap_collection

	proxy_laps(traps, image_loader)


def evaluate_current_cube():
	db = Loader.load()

	from misccube.cubeload.load import CubeLoader
	from magiccube.laps.traps.trap import IntentionType
	from magiccube.laps.traps.tree.printingtree import AllNode, PrintingNode

	cube_loader = CubeLoader(db)

	cube_loader.check_and_update()

	cube = cube_loader.load()

	fetcher = ConstrainedCubeableFetcher(db)

	constrained_nodes = fetcher.fetch()

	garbage_traps = [trap for trap in cube.traps if trap.intention_type == IntentionType.GARBAGE]

	index_map = {} #type: t.Dict[PrintingNode, int]

	for index, trap in enumerate(garbage_traps):
		for child in trap.node.children:
			index_map[child if isinstance(child, PrintingNode) else AllNode((child,))] = index

	traps = [[] for _ in range(len(garbage_traps))]

	for node in constrained_nodes:
		try:
			traps[index_map[node.node]].append(node)
		except KeyError as e:
			if isinstance(node.node, AllNode):
				traps[index_map[AllNode((node.node.children.__iter__().__next__(),))]].append(node)
			else:
				raise e

	distribution = TrapDistribution(
		traps = traps
	)

	distributor = Distributor(constrained_nodes, len(garbage_traps), GROUP_WEIGHTS).evaluate(0)

	print(
		algorithm.distribution_score(
			distribution,
			distributor,
		)
	)

	# vs: .9625556483124305


if __name__ == '__main__':
	main()
import typing as t

import os

from multiset import Multiset

from promise import Promise

from proxypdf.write import save_proxy_pdf

from mtgorp.models.persistent.printing import Printing

from mtgimg.load import Loader as ImageLoader

from magiccube.collections.cube import Cube
from magiccube.laps.lap import Lap

from misccube.cubeload.load import CubeLoader


class LapProxyer(object):
	OUT_DIR = 'out'

	def __init__(self, cube_loader: CubeLoader, margin_size: float = .1, card_margin_size: float = .05):
		self._cube_loader = cube_loader
		self.margin_size = margin_size
		self.card_margin_size = card_margin_size

		self._image_loader = ImageLoader()

	def proxy_cube(self, cube: Cube, file_name: t.AnyStr) -> None:

		promises = tuple(
			self._image_loader.get_image(lap)
			for lap in
			cube.laps
		)

		images = Promise.all(
			promises
		).get()

		save_proxy_pdf(
			file = file_name,
			images = images,
			margin_size = self.margin_size,
			card_margin_size = self.card_margin_size,
		)

	def pdf_all_images(self) -> None:
		self.proxy_cube(
			cube = self._cube_loader.load(),
			file_name = os.path.join(self.OUT_DIR, 'all_laps.pdf')
		)
	
	def pdf_difference_images(
		self
	) -> t.Tuple[t.Collection[Printing], t.Collection[Printing], t.Collection[Lap], t.Collection[Lap]]:

		cubes = self._cube_loader.all_cubes()

		new_cube, old_cube = next(cubes, Cube()), next(cubes, Cube())

		positive_difference, negative_difference = new_cube - old_cube, old_cube - new_cube

		new_printings, old_printings = Multiset(new_cube.all_printings), Multiset(old_cube.all_printings)

		additional_printings = new_printings.difference(old_printings)

		removed_printings = old_printings.difference(new_printings)

		self.proxy_cube(
			cube = positive_difference,
			file_name = os.path.join(self.OUT_DIR, 'new_laps.pdf')
		)

		self.proxy_cube(
			cube = negative_difference,
			file_name = os.path.join(self.OUT_DIR, 'removed_laps.pdf')
		)

		return additional_printings, removed_printings, positive_difference, negative_difference


def run():
	from mtgorp.db.load import Loader

	db = Loader.load()

	cube_loader = CubeLoader(db)

	cube_loader.check_and_update()

	lapper = LapProxyer(
		cube_loader,
		margin_size = .25,
		card_margin_size = .0,
	)

	lapper.pdf_all_images()

	additional_printings, removed_printings, positive_difference, negative_difference = lapper.pdf_difference_images()

	print('Additional printings: ' + str(additional_printings))
	print('Removed printings: ' + str(removed_printings))
	print('Positive difference: ' + str(positive_difference))
	print('Negative difference: ' + str(negative_difference))


if __name__ == '__main__':
	run()
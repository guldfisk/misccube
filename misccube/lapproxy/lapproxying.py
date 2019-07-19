import typing as t

import os

from promise import Promise

from proxypdf.write import save_proxy_pdf

from mtgimg.load import Loader as ImageLoader

from magiccube.collections.cube import Cube
from magiccube.collections.delta import CubeDelta

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
	
	def difference_report(
		self
	) -> CubeDelta:

		cubes = self._cube_loader.all_cubes()

		current_cube, _ = next(cubes, Cube())
		previous_cube, _ = next(cubes, Cube())

		delta = CubeDelta(previous_cube, current_cube)

		self.proxy_cube(
			cube = delta.new_cubeables,
			file_name = os.path.join(self.OUT_DIR, 'new_laps.pdf')
		)

		self.proxy_cube(
			cube = delta.removed_cubeables,
			file_name = os.path.join(self.OUT_DIR, 'removed_laps.pdf')
		)

		return delta


def run():
	from mtgorp.db.load import Loader

	db = Loader.load()

	cube_loader = CubeLoader(db)

	cube_loader.check_and_update()

	lapper = LapProxyer(
		cube_loader,
		margin_size = .7,
		card_margin_size = .05,
	)

	lapper.pdf_all_images()

	delta = lapper.difference_report()

	print(delta.report)


def proxies_from_strings():
	from mtgorp.db.load import Loader

	db = Loader.load()

	strings = [
		'(2# Vengevine|193556); Boil|4805; Decimate|31798; Kuldotha Forgemaster|215098; Shelldock Isle|146178; Underground River|11545',
		'(3# Rite of Flame|121217); Crop Rotation|12432; Darksteel Citadel|45415; Everflowing Chalice|198374; Serum Visions|50145',
	]

	from magiccube.laps.traps.tree.parse import PrintingTreeParser
	from magiccube.laps.traps.trap import Trap

	parser = PrintingTreeParser(db)
	image_loader = ImageLoader()

	traps = [
		Trap(
			parser.parse(s)
		)
		for s in
		strings
	]

	promises = [
		image_loader.get_image(trap)
		for trap in
		traps
	]

	images = Promise.all(
		promises
	).get()

	save_proxy_pdf(
		file=os.path.join('out', 'select_traps.pdf'),
		images=images,
		margin_size=.7,
		card_margin_size=.05,
	)




if __name__ == '__main__':
	proxies_from_strings()
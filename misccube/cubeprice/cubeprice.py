import requests as r

from mtgorp.db.load import Loader as DbLoader
from mtgorp.models.persistent.printing import Printing

from magiccube.collections.cube import Cube

from misccube.cubeload.load import CubeLoader


class Pricer(object):

	@classmethod
	def price_printing(cls, printing: Printing) -> float:
		uri = f'https://api.scryfall.com/cards/multiverse/{printing.id}'

		remote_card = r.get(uri)

		if not remote_card.ok:
			print('rip', printing)
			return 0.

		json = remote_card.json()

		try:
			return float( json['eur'] )

		except KeyError:
			print('rip', printing)
			return 0.

	@classmethod
	def price(cls, cube: Cube) -> int:
		return sum(
			cls.price_printing(
				printing
			)
			for printing in
			cube.all_printings
		)


def test():

	db_loader = DbLoader()

	db = db_loader.load()

	cube_loader =  CubeLoader(db)

	cube = cube_loader.load()

	print(
		Pricer.price(
			cube
		)
	)


if __name__ == '__main__':
	test()
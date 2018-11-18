

from mtgorp.db.load import Loader
from mtgorp.tools.parsing.search.parse import SearchParser, ParseException


from misccube.cubeload.load import CubeLoader


def run():

	db = Loader.load()

	cube_loader = CubeLoader(db)

	cube = cube_loader.load()

	cardboards = set(printing.cardboard for printing in cube.all_printings)

	search_parser = SearchParser(db)

	print(f'cube loaded, {len(cardboards)} unique cardboards')

	while True:
		query = input(': ')

		try:
			pattern = search_parser.parse(query)
		except ParseException as e:
			print(f'Invalid query "{e}"')
			continue

		results = list(pattern.matches(cardboards))

		print(
			'\n'
				.join(
					sorted(
						(cardboard.name for cardboard in results),
					)
				)
			+ f'\n\n-------\n{len(results)} result{"" if len(results) == 1 else "s"}'
		)


if __name__ == '__main__':
	run()
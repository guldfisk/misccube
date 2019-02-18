
from multiset import Multiset

from mtgorp.db.load import Loader

from misccube.cubeload.load import CubeLoader



def count():

	db = Loader.load()

	cube = CubeLoader(db).load()

	# for printing in (printing for printing in set(cube.all_printings) if printing.front_face.artist.name=='Eric Deschamps'):
	# 	print(printing)

	artists = Multiset(
		printing.front_face.artist
		for printing in
		set(cube.all_printings)
	)

	for artist, multiplicity in sorted(artists.items(), key=lambda vs: vs[1]):
		print(artist, multiplicity)

	# print(sum(multiplicity for artist, multiplicity in sorted(artists.items(), key=lambda vs: vs[1])))




if __name__ == '__main__':
	count()
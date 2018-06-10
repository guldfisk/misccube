
import os
from itertools import chain

from matplotlib import pyplot as plt
import numpy as np

from mtgorp.db.load import Loader

from misccube.cubeload.load import CubeLoader


def run():

	# matplotlib.rc('font', size=20)

	out_path = os.path.join('out', 'print_distr')

	db = Loader.load()
	cube_loader = CubeLoader(db)
	cube = cube_loader.load()

	original_printings_printings = [
		printing.cardboard.original_printing
		for printing in
		cube.printings
	]

	original_printings_garbage = [
		printing.cardboard.original_printing
		for printing in
		cube.garbage_printings
	]

	original_expansions = (
		expansion
		for expansion in
		db.expansions.values()
		if any(
			printing.cardboard.original_printing.expansion == expansion
			for printing in
			expansion.printings
		)
	)

	expansions_chronologically = sorted(
		original_expansions,
		key =lambda expansion:
			expansion.release_date
	)

	expansion_occurrences_printings = {
		expansion: 0
		for expansion in
		db.expansions.values()
	}

	expansion_occurrences_garbage = {
		expansion: 0
		for expansion in
		db.expansions.values()
	}

	for printing in original_printings_printings:
		expansion_occurrences_printings[printing.expansion] += 1

	for printing in original_printings_garbage:
		expansion_occurrences_garbage[printing.expansion] += 1

	figure, axis = plt.subplots()

	_range = np.arange(len(expansions_chronologically))

	sum_rects = axis.bar(
		_range,
		[
			expansion_occurrences_printings[expansion] + expansion_occurrences_garbage[expansion]
			for expansion in
			expansions_chronologically
		],
		width = .8,
		color = 'r',
	)

	printings_rects = axis.bar(
		_range,
		[
			expansion_occurrences_printings[expansion]
			for expansion in
			expansions_chronologically
		],
		width = .8,
		color = 'b',
	)

	axis.set_xticks(_range)
	axis.set_xticklabels(
		[
			str(expansion.code)
			for expansion in
			expansions_chronologically
		],
		rotation = 90,
	)

	axis.legend((printings_rects, sum_rects), ('Non-garbage', 'Garbage'))

	axis.set_ylabel('Qty. original printings')
	axis.set_xlabel('Expansion chronologically')

	axis.set_title('Cube cardboards original printings distribution')

	figure.set_figwidth(40)
	figure.set_figheight(10)

	plt.savefig(out_path)

if __name__ == '__main__':
	run()
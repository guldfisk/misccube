import typing as t

import re

from sheetclient.client import GoogleSheetClient

from mtgorp.db.database import CardDatabase

from magiccube.laps.traps.tree.parse import PrintingTreeParser, PrintingTreeParserException
from magiccube.laps.traps.trap import Trap
from magiccube.laps.traps.tree.printingtree import PrintingNode

from misccube.trapification.algorithm import ConstrainedNode, TrapDistribution
from misccube import values



class ConstrainedCubeablesFetchException(Exception):
	pass



class ConstrainedCubeableFetcher(object):
	SHEET_NAME = 'trapables'

	_value_value_map = {
		1:  1,
		2: 5,
		3: 15,
		4: 30,
		5: 60,
	}

	_legal_groups = {
		'drawgo',
		'mud',
		'post',
		'midrange',
		'mill',
		'reanimate',
		'burn',
		'hatebear',
		'removal',
		'lock',
		'yardvalue',
		'ld',
		'storm',
		'tezz',
		'lands',
		'shatter',
		'bounce',
		'shadow',
		'stifle',
		'beat',
		'cheat',
		'pox',
		'counter',
		'discard',
		'cantrip',
	}

	def __init__(self, db: CardDatabase, document_id: str = values.DOCUMENT_ID):
		self._db = db

		self._printing_tree_parser = PrintingTreeParser(db)
		self._sheet_client = GoogleSheetClient(document_id)

	def _parse_groups(self, s: str) -> t.FrozenSet[str]:
		groups = []

		if s:
			for group in s.split(','):
				group = group.rstrip().lstrip()

				if not group in self._legal_groups:
					raise ConstrainedCubeablesFetchException(f'"{group}" not a legal group')

				groups.append(group)

		return frozenset(groups)

	def fetch(self) -> t.List[ConstrainedNode]:
		exceptions = []
		constrained_cubeables = []

		for row in self._sheet_client.read_sheet(
			self.SHEET_NAME,
			start_column = 1,
			start_row = 4,
			end_column = 4,
			end_row = 500,
		):
			amount_cell, printings_cell, value_cell = row[:3]
			groups_cell = row[3] if len(row) > 3 else ''

			try:
				amount = int(amount_cell)
			except ValueError:
				amount = 1

			try:
				node = self._printing_tree_parser.parse(printings_cell)
			except PrintingTreeParserException as e:
				exceptions.append(e)
				continue

			try:
				value = self._value_value_map[int(value_cell)]
			except (ValueError, KeyError) as e:
				exceptions.append(e)
				continue

			try:
				groups = self._parse_groups(groups_cell)
			except ConstrainedCubeablesFetchException as e:
				exceptions.append(e)
				continue

			node = ConstrainedNode(
				node = node,
				value = value,
				groups = groups,
			)

			for _ in range(amount):
				constrained_cubeables.append(node)

		if exceptions:
			raise ConstrainedCubeablesFetchException(exceptions)

		return constrained_cubeables


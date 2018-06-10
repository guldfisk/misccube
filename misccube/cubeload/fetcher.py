import typing as t

import re

import numpy as np

import requests

from mtgorp.db.database import CardDatabase
from mtgorp.models.persistent.printing import Printing

from magiccube.collections.cube import Cube
from magiccube.laps.traps.tree.parse import PrintingTreeParser, PrintingTreeParserException
from magiccube.laps.traps.trap import Trap
from magiccube.laps.tickets.ticket import Ticket


DOCUMENT_ID = '1zhZuAAAYZk_f3lCsi0oFXXiRh5jiSMydurKnMYR6HJ8'
SHEET_ID = '690106443'


def tsv_to_ndarray(value: str, separator: str = '\t', line_separator: str = '\n') -> np.ndarray:
	rows = value.split(line_separator)

	matrix = [
		row.split(separator)
		for row in
		rows
	]

	row_length = max(map(len, matrix))

	for row in matrix:
		for i in range(row_length - len(row)):
			row.append('')

	return np.asarray(matrix, dtype=str)


class CubeFetchException(Exception):
	pass


class CubeParseException(CubeFetchException):
	pass


class CubeFetcher(object):

	def __init__(self, db: CardDatabase, document_id: str = DOCUMENT_ID, sheet_id: str = SHEET_ID):
		self._db = db
		self._printing_tree_parser = PrintingTreeParser(self._db)
		self._document_id = document_id
		self._sheet_id = sheet_id
		self._printing_matcher = re.compile(" *([\\w\\-': ,/]+)\\|([A-Z0-9_]+)")
		self._ticket_matcher = re.compile("([\\w ]+):(.*)")
		self._id_matcher = re.compile('\\d+$')

	def _get_tsv(self) -> np.ndarray:
		try:
			response = requests.get(
				'https://docs.google.com/spreadsheets/d/{}/pub?gid={}&single=true&output=tsv'.format(
					self._document_id,
					self._sheet_id,
				)
			)
		except requests.ConnectionError as e:
			raise CubeFetchException(e)

		if not response.ok:
			raise CubeFetchException(response.status_code)

		return tsv_to_ndarray(
			response
				.content
				.decode('UTF-8')
		)

	def _parse_lap_cell(self, cell: str) -> Trap:
		try:
			return Trap(self._printing_tree_parser.parse(cell))
		except PrintingTreeParserException as e:
			raise CubeParseException(e)

	def _parse_printing(self, s: str) -> Printing:
		m = self._printing_matcher.match(s)
		if not m:
			raise CubeParseException(f'Invalid printing "{s}"')

		if self._id_matcher.match(m.group(2)):
			try:
				return self._db.printings[int(m.group(2))]
			except KeyError:
				raise CubeParseException(f'Invalid printing id "{m.group(2)}"')

		try:
			cardboard = self._db.cardboards[m.group(1)]
		except KeyError:
			raise CubeParseException(f'Invalid cardboard "{m.group(1)}"')

		try:
			return cardboard.from_expansion(m.group(2))
		except KeyError:
			raise CubeParseException(f'Invalid expansion "{m.group(2)}" for cardboard "{cardboard}"')

	def _parse_ticket(self, s: str) -> Ticket:
		m = self._ticket_matcher.match(s)
		if not m:
			raise CubeParseException(f'Invalid ticket "{s}"')

		return Ticket(
			[
				self._parse_printing(sm.group())
				for sm in
				self._printing_matcher.finditer(m.group(2))
			],
			m.group(1)
		)

	def _construct_cube(self, tsv: np.ndarray) -> Cube:
		traps = []
		printings = []
		tickets = []
		for column in tsv.T:
			if column[0] == 'TRAPS':
				for cell in column[2:]:
					if cell:
						traps.append(self._parse_lap_cell(cell))
			elif column[0] in ('W', 'U', 'B', 'R', 'G', 'HYBRID', 'GOLD', 'COLORLESS', 'LAND'):
				for cell in column[2:]:
					if cell:
						printings.append(self._parse_printing(cell))
			elif column[0] == 'TICKETS':
				for cell in column[2:]:
					if cell:
						tickets.append(self._parse_ticket(cell))

		return Cube(printings, traps, tickets)

	def fetch_cube(self) -> Cube:
		return self._construct_cube(
			self._get_tsv()
		)

import typing as t

import re
import itertools

import numpy as np

from sheetclient.client import GoogleSheetClient

from mtgorp.db.database import CardDatabase
from mtgorp.models.persistent.printing import Printing

from magiccube.collections.cube import Cube, Cubeable
from magiccube.laps.traps.tree.parse import PrintingTreeParser, PrintingTreeParserException
from magiccube.laps.traps.trap import Trap
from magiccube.laps.tickets.ticket import Ticket
from magiccube.laps.purples.purple import Purple

from misccube import values


T = t.TypeVar('T', bound = Cubeable)


SHEET_ID = '690106443'
SHEET_NAME = 'liste'


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

    def __init__(self, db: CardDatabase, document_id: str = values.DOCUMENT_ID, sheet_id: str = SHEET_ID):
        self._db = db
        self._printing_tree_parser = PrintingTreeParser(self._db)
        self._document_id = document_id
        self._sheet_id = sheet_id
        self._printing_matcher = re.compile("\\s*([\\w\\-': ,/]+)\\|([A-Z0-9_]+)")
        self._ticket_matcher = re.compile("([\\w ]+):(.*)")
        self._id_matcher = re.compile('\\d+$')

        self._sheet_client = GoogleSheetClient(document_id)

    def _get_cells(self) -> t.List[t.List[str]]:
        return self._sheet_client.read_sheet(
            sheet_name = SHEET_NAME,
            start_column = 1,
            start_row = 1,
            end_column = 50,
            end_row = 300,
            major_dimension = 'COLUMNS',
        )

    def _parse_trap_cell(self, cell: str, arg: t.Optional[str] = None) -> Trap:
        try:
            return Trap(
                node = self._printing_tree_parser.parse(cell),
                intention_type = (
                    Trap.IntentionType.GARBAGE
                    if arg is None or arg == '' else
                    Trap.IntentionType(arg.split('-')[0])
                ),
            )
        except (PrintingTreeParserException, AttributeError) as e:
            raise CubeParseException(e)

    def _parse_printing(self, s: str, arg: str = None) -> Printing:
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

    def _parse_ticket(self, s: str, arg: str = None) -> Ticket:
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

    def _parse_purple(self, s: str, arg: str = None) -> Purple:
        return Purple(s)

    def _construct_cube(self, tsv: t.List[t.List[str]]) -> Cube:
        traps = []
        printings = []
        tickets = []
        purples = []

        exceptions = [] #type: t.List[t.Tuple[str, Exception]]

        def _parse_all_cells(
            cells: t.Iterable[str],
            arguments: t.Iterable[str],
            parser: t.Callable[[str, str], T],
        ) -> t.Iterable[T]:

            for _cell, arg in itertools.zip_longest(cells, arguments, fillvalue=''):
                if _cell:
                    try:
                        yield parser(_cell, arg)
                    except CubeParseException as e:
                        exceptions.append((_cell, e))

        for column, args in itertools.zip_longest(tsv, tsv[1:], fillvalue=['' for _ in range(len(tsv[-1]))]):

            if column[0] == 'TRAPS':
                traps.extend(list(_parse_all_cells(column[2:], args[2:], self._parse_trap_cell)))

            if column[0] == 'GARBAGE_TRAPS':
                traps.extend(list(_parse_all_cells(column[2:], (), self._parse_trap_cell)))

            elif column[0] in ('W', 'U', 'B', 'R', 'G', 'HYBRID', 'GOLD', 'COLORLESS', 'LAND'):
                printings.extend(_parse_all_cells(column[2:], args[2:], self._parse_printing))

            elif column[0] == 'TICKETS':
                tickets = list(_parse_all_cells(column[2:], args[2:], self._parse_ticket))

            elif column[0] == 'PURPLES':
                purples = list(_parse_all_cells(column[2:], args[2:], self._parse_purple))

        if exceptions:
            raise CubeParseException(exceptions)

        return Cube(
            itertools.chain(printings, traps, tickets, purples)
        )

    def fetch_cube(self) -> Cube:
        return self._construct_cube(
            self._get_cells()
        )

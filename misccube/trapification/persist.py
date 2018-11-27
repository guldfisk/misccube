
import typing as t

import datetime
import os

from mtgorp.db.database import CardDatabase
from mtgorp.models.serilization.serializeable import Serializeable, serialization_model, Inflator
from mtgorp.models.serilization.strategies.jsonid import JsonId
from mtgorp.utilities.containers import HashableMultiset

from magiccube.laps.traps.trap import Trap

from misccube import paths


class TrapCollection(Serializeable):
	
	def __init__(self, traps: t.Iterable[Trap]):
		self._traps = HashableMultiset(traps)

	def serialize(self) -> serialization_model:
		return {
			'traps': self._traps,
		}

	@classmethod
	def deserialize(cls, value: serialization_model, inflator: Inflator) -> 'Serializeable':
		return cls(value['traps'])

	def __hash__(self) -> int:
		return hash(self._traps)

	def __eq__(self, other: object) -> bool:
		return (
			isinstance(other, self.__class__)
			and self._traps == other._traps
		)


class TrapCollectionPersistor(object):
	
	_OUT_DIR = os.path.join(paths.OUT_DIR, 'trap_collections')
	TIMESTAMP_FORMAT = '%y_%m_%d_%H_%M_%S'

	def __init__(self, db: CardDatabase):
		self._db = db
		self._strategy = JsonId(self._db)
		
	def persist(self, traps: t.Iterable[Trap]):
		if not os.path.exists(self._OUT_DIR):
			os.makedirs(self._OUT_DIR)
			
		with open(
			os.path.join(
				self._OUT_DIR,
				datetime.datetime.strftime(
					datetime.datetime.today(),
					self.TIMESTAMP_FORMAT,
				),
			) + '.json',
			'w',
		) as f:
			
			f.write(
				JsonId.serialize(
					TrapCollection(
						traps
					)
				)
			)
			
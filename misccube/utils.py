
import typing as t

import random


T = t.TypeVar('T')


def choice(collection: t.Collection[T]) -> T:
	if not hasattr(collection, '__getitem__'):
		return random.choice(list(collection))

	return random.choice(collection)
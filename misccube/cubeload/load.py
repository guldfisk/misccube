import typing as t

import os
import datetime

from mtgorp.db.database import CardDatabase
from mtgorp.models.serilization.strategies.jsonid import JsonId

from magiccube.collections.cube import Cube

from misccube import paths
from misccube.cubeload.fetcher import CubeFetcher


class CubeLoadException(Exception):
    pass


class CubeLoader(object):

    LOCAL_CUBES_PATH = os.path.join(paths.APP_DATA_PATH, 'cubes')
    TIMESTAMP_FORMAT = '%d_%m_%y_%H_%M_%S'

    def __init__(self, db: CardDatabase):
        self._db = db
        self._fetcher = CubeFetcher(self._db)
        self._strategy = JsonId(self._db)

    def get_local_cube_paths(self) -> t.Iterator[t.Tuple[str, datetime.datetime]]:
        if not os.path.exists(self.LOCAL_CUBES_PATH):
            os.makedirs(self.LOCAL_CUBES_PATH)

        cubes = os.listdir(self.LOCAL_CUBES_PATH)

        if not cubes:
            return

        names_times = [] #type: t.List[t.Tuple[str, datetime.datetime]]

        for cube in cubes:
            try:
                names_times.append(
                    (
                        cube,
                        datetime.datetime.strptime(cube, self.TIMESTAMP_FORMAT),
                    )
                )
            except ValueError:
                pass

        if not names_times:
            return

        sorted_pairs = sorted(names_times, key=lambda item: item[1], reverse = True)

        for name, time in sorted_pairs:
            yield os.path.join(self.LOCAL_CUBES_PATH, name), time

    def _get_all_local_cubes(self) -> t.Iterator[t.Tuple[Cube, datetime.datetime]]:
        for path, time in self.get_local_cube_paths():
            with open(path, 'r') as f:
                yield self._strategy.deserialize(Cube, f.read()), time

    def _get_current_local_cube(self) -> t.Optional[Cube]:
        try:
            return self._get_all_local_cubes().__next__()[0]
        except StopIteration:
            return None
        
    @classmethod
    def _persist_cube(cls, cube: Cube) -> None:
        if not os.path.exists(cls.LOCAL_CUBES_PATH):
            os.makedirs(cls.LOCAL_CUBES_PATH)

        with open(
            os.path.join(
                cls.LOCAL_CUBES_PATH,
                datetime.datetime.strftime(
                    datetime.datetime.today(),
                    cls.TIMESTAMP_FORMAT,
                ),
            ),
            'w',
        ) as f:
            f.write(JsonId.serialize(cube))

    def rollback(self) -> None:
        try:
            path, time = self.get_local_cube_paths().__next__()
        except StopIteration:
            return

        os.remove(path)

    def check_and_update(self) -> bool:
        local_cube = self._get_current_local_cube()

        remote_cube = self._fetcher.fetch_cube()

        if local_cube is None or local_cube != remote_cube:
            self._persist_cube(remote_cube)
            return True

        return False

    def load(self) -> Cube:
        cube = self._get_current_local_cube()

        if cube is None:
            if not self.check_and_update():
                raise CubeLoadException()
            return self._get_current_local_cube()

        return cube
    
    def all_cubes(self) -> t.Iterator[t.Tuple[Cube, datetime.datetime]]:
        if self._get_current_local_cube() is None:
            if not self.check_and_update():
                raise CubeLoadException()
            
        return self._get_all_local_cubes()

import typing as t

from magiccube.collections.laps import TrapCollection
from magiccube.collections.meta import MetaCube
from magiccube.update.report import UpdateReport
from misccube.cubeload.load import CubeLoader
from misccube.trapification.fetch import ConstrainedNodeFetcher
from yeetlong.multiset import Multiset

from mtgorp.db.load import Loader

from mtgorp.models.serilization.serializeable import Serializeable, serialization_model, Inflator
from mtgorp.models.serilization.strategies.jsonid import JsonId

from magiccube.laps.traps.tree.printingtree import BorderedNode, PrintingNode, AllNode, AnyNode
from magiccube.laps.traps.trap import Trap, IntentionType
from magiccube.laps.traps.tree.parse import PrintingTreeParser
from magiccube.collections.cube import Cube
from magiccube.collections.nodecollection import NodeCollection, NodesDeltaOperation, ConstrainedNode, GroupMap
from magiccube.collections.delta import CubeDeltaOperation
from magiccube.update.cubeupdate import CubePatch, CubeUpdater

_GROUP_WEIGHTS = {
    'WHITE': 1,
    'BLUE': 1.5,
    'BLACK': 1,
    'RED': 1,
    'GREEN': 1,
    'drawgo': 3,
    'mud': 3,
    'post': 4,
    'midrange': 2,
    'mill': 4,
    'reanimate': 4,
    'burn': 4,
    'hatebear': 2,
    'removal': 1,
    'lock': 3,
    'yardvalue': 3,
    'ld': 3,
    'storm': 4,
    'tezz': 3,
    'lands': 3,
    'shatter': 3,
    'bounce': 3,
    'shadow': 4,
    'stifle': 4,
    'beat': 1,
    'cheat': 4,
    'pox': 3,
    'counter': 3,
    'discard': 2,
    'cantrip': 4,
    'balance': 3,
    'stasis': 4,
    'standstill': 3,
    'whitehate': 4,
    'bluehate': 4,
    'blackhate': 4,
    'redhate': 4,
    'greenhate': 4,
    'antiwaste': 4,
    'delirium': 3,
    'sacvalue': 2,
    'lowtoughnesshate': 4,
    'armageddon': 4,
    'stax': 3,
    'bloom': 3,
    'weldingjar': 3,
    'drawhate': 4,
    'pluscard': 3,
    'ramp': 3,
    'devoteddruid': 4,
    'fetchhate': 4,
    'dragon': 2,
    'company': 2,
    'naturalorder': 3,
    'flash': 3,
    'wincon': 3,
    'vial': 4,
    # lands
    'fixing': 3,
    'colorlessvalue': 1,
    'fetchable': 2,
    'indestructable': 4,
    'legendarymatters': 1,
    'sol': 3,
    'manland': 4,
    'storage': 3,
    'croprotate': 3,
    'dnt': 3,
    'equipment': 4,
    'livingdeath': 3,
    'eggskci': 3,
    'hightide': 3,
    'fatty': 3,
    'walker': 4,
    'blink': 2,
    'miracles': 3,
    'city': 4,
    'wrath': 2,
    'landtax': 4,
    'discardvalue': 2,
    'edict': 2,
    'phoenix': 4,
    'enchantress': 2,
    'dork': 2,
    'tinker': 3,
    'highpowerhate': 2,
    'affinity': 3,
    'academy': 4,
    'stompy': 2,
    'shardless': 3,
    'lanterns': 3,
    'depths': 3,
    'survival': 2,
    'landstill': 2,
    'moat': 4,
    'combo': 3,
    'kite': 3,
    'haste': 3,
    'fog': 3,
    'threat': 4,
}


def test():
    db = Loader.load()
    strategy = JsonId(db)
    cube = CubeLoader(db).load()

    constrained_nodes = NodeCollection(
        ConstrainedNodeFetcher(db).fetch_garbage()
    )

    groups = GroupMap(_GROUP_WEIGHTS)

    # s = '{"cube_delta": {}, "nodes_delta": {"nodes": []}}'
    # patch = strategy.deserialize(CubePatch, s)

    patch = CubePatch(
        CubeDeltaOperation(
            {
                db.cardboards['Brainstorm'].from_expansion('ICE'): -1,
                db.cardboards['Brainstorm'].from_expansion('EMA'): 1,
                # Trap(
                #     AllNode(
                #         (
                #             db.cardboards['Brainstorm'].from_expansion('ICE'),
                #             db.cardboards['Web'].from_expansion('LEA'),
                #         )
                #     ),
                #     intention_type=IntentionType.SYNERGY,
                # ): 2
            }
        ),
        NodesDeltaOperation(
            {
                # ConstrainedNode(
                #     node = AllNode(
                #         (
                #             db.cardboards['Web'].from_expansion('LEA'),
                #         )
                #     ),
                #     groups = ['ok', 'lmao'],
                #     value = 2,
                # ): 1,
                ConstrainedNode(
                    node=AllNode(
                        (
                            db.cardboards['Brainstorm'].from_expansion('ICE'),
                            db.cardboards['Web'].from_expansion('LEA'),
                        )
                    ),
                    groups=['lolHAHA'],
                    value=1,
                ): 1,
            }
        )
    )

    print(patch)

    meta_cube = MetaCube(cube, constrained_nodes, groups)

    verbose_patch = patch.as_verbose(meta_cube)

    print(verbose_patch)

    updater = CubeUpdater(meta_cube, patch)

    print(updater)

    report = UpdateReport(updater)

    for notification in report.notifications:
        print(notification.title + '\n' + notification.content + '\n\n')

    # printing_tree_parser = PrintingTreeParser(db)
    #
    # trap = Trap(
    #     printing_tree_parser.parse(s)
    # )
    #
    # print(trap)
    #
    # print(strategy.serialize(trap))
    # print(strategy.deserialize(Trap, strategy.serialize(trap)))
    # print(strategy.deserialize(Trap, strategy.serialize(trap)) == trap)


if __name__ == '__main__':
    test()

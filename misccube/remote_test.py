import os
import itertools
from collections import defaultdict

from yeetlong.multiset import Multiset, FrozenMultiset

from evolution import model

from misccube.trapification.persist import DistributionModel

from mtgorp.models.serilization.strategies.jsonid import JsonId

from magiccube.laps.traps.trap import IntentionType
from magiccube.collections.cube import Cube
from magiccube.collections.delta import CubeDelta, CubeDeltaOperation
from mtgorp.managejson.update import check_and_update
from mtgorp.db.create import update_database
from mtgorp.db.load import Loader

from magiccube.laps.traps.distribute import algorithm
from magiccube.collections.nodecollection import NodeCollection

from misccube.cubeload.load import CubeLoader
from misccube.trapification.fetch import ConstrainedNodeFetcher
from misccube import paths


_GROUP_MAP = {
    key: value / 4
    for key, value in
    {
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
    }.items()
}

_value_value_map = {
    key: value / 55
    for key, value in
    {
        0: 0,
        1: 1,
        2: 5,
        3: 15,
        4: 30,
        5: 55,
    }.items()
}


def test():
    db = Loader().load()
    # constrained_nodes = NodeCollection(
    #     ConstrainedNodeFetcher(db).fetch_all()
    # )

    strategy = JsonId(db)

    with open(os.path.join(paths.OUT_DIR, 'constrained_nodes.json'), 'r') as f:
        # f.write(JsonId.serialize(constrained_nodes))
        constrained_nodes = strategy.deserialize(NodeCollection, f.read())
        # constrained_nodes._nodes = FrozenMultiset(frozenset(constrained_nodes._nodes))


    for node in constrained_nodes.nodes.distinct_elements():
        node._value = _value_value_map[node._value]

    distribution_nodes = list(map(algorithm.DistributionNode, constrained_nodes))

    trap_amount = 118

    constraints = model.ConstraintSet(
        (
            (
                algorithm.SizeHomogeneityConstraint(
                    distribution_nodes,
                    trap_amount,
                ),
                1,
            ),
            (
                algorithm.ValueDistributionHomogeneityConstraint(
                    distribution_nodes,
                    trap_amount,
                ),
                2,
            ),
            (
                algorithm.GroupExclusivityConstraint(
                    distribution_nodes,
                    trap_amount,
                    _GROUP_MAP,
                ),
                2,
            ),
        )
    )

    # with open(os.path.join(paths.OUT_DIR, 'old_distribution.json'), 'r') as f:
    #     competitor = strategy.deserialize(DistributionModel, f.read())
    #
    # for trap in competitor.traps:
    #     for node in trap:
    #         node._value = _value_value_map[node._value]
    #
    # constrained_nodes_set = FrozenMultiset(constrained_nodes)
    # competitor_constrained_nodes_set = FrozenMultiset(
    #     itertools.chain(
    #         *competitor.traps
    #     )
    # )
    #
    # s = Multiset(node.node for node in constrained_nodes_set)
    # ss = Multiset(node.node for node in competitor_constrained_nodes_set)
    #
    # competitor_distribution = algorithm.TrapDistribution(
    #     traps = [
    #         [
    #             algorithm.DistributionNode(node)
    #             for node in
    #             trap
    #         ]
    #         for trap in
    #         competitor.traps
    #     ]
    # )
    #
    # print('competitor score', constraints.score(competitor_distribution))

    distributor = algorithm.Distributor(
        nodes = constrained_nodes,
        trap_amount = trap_amount,
        initial_population_size = 300,
        group_weights = _GROUP_MAP,
        constraints = constraints,
    )

    distributor.spawn_generations(1200)

    distributor.show_plot()

    fittest = distributor.fittest()

    print(fittest)
    print(fittest.fitness)


if __name__ == '__main__':
    test()
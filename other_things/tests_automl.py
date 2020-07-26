"""Test."""

import numpy as np

import pjautoml.cs.operator.free.map as pja
import pjdata.content.specialdata as sd
from pjautoml.cs.operator.datadriven.optimization.modelfree.best import Best
from pjautoml.cs.operator.datadriven.optimization.modelfree.random import RandomSearch
from pjautoml.cs.operator.free.chain import Chain
from pjautoml.cs.operator.free.sample import Sample
from pjautoml.cs.operator.free.select import Select
from pjdata.aux.util import _
from pjml.abs.mixin.timing import withTiming
from pjml.data.communication.cache import Cache
from pjml.data.communication.report import Report
from pjml.data.evaluation.metric import Metric
from pjml.data.evaluation.split import Split, TrSplit, TsSplit
from pjml.data.flow.file import File
from pjml.operator.pipeline import Pipeline
from pjml.stream.expand.partition import Partition
from pjml.stream.reduce.reduce import Reduce
from pjml.stream.reduce.summ import Summ
from pjml.stream.transform.map import Map
from pjpy.modeling.supervised.classifier.dt import DT
from pjpy.modeling.supervised.classifier.svmc import SVMC
from pjpy.processing.feature.binarize import Binarize
from pjpy.processing.feature.reductor.pca import PCA
from pjpy.processing.feature.scaler.minmax import MinMax


def printable_test():
    """toy test."""
    dt_tree = DT()
    dt_tree.disable_pretty_printing()
    print(repr(dt_tree))
    print(dt_tree)
    dt_tree.enable_pretty_printing()
    print()
    print(repr(dt_tree))
    print(dt_tree)


def test_tsvmc(arq="iris.arff"):
    cs = File(arq)
    pipe = Pipeline(File(arq), SVMC())
    train, test = pipe.dual_transform()
    print("Train..............\n", train)
    print("Test..........\n", test)


def test_split(arq="iris.arff"):
    pipe = Pipeline(File(arq), Split(), SVMC())
    train, test = pipe.dual_transform()
    print("Train..............\n", str(train))
    print("Test..........\n", str(test))


def test_metric(arq="iris.arff"):
    pipe = Pipeline(File(arq), Split(), SVMC(), Metric(enhance=False))
    train, test = pipe.dual_transform()
    print("Train..............\n", train)
    print("Test..........\n", test)


def test_pca(arq="iris.arff"):
    pipe = Pipeline(File(arq), Split(), PCA(), SVMC(), Metric())
    train, test = pipe.dual_transform()
    print("Train..............\n", _.m(_.name, train.history))
    print("Test..........\n", _.m(_.name, test.history))


def test_partition(arq="iris.arff"):
    pipe = Pipeline(
        File(arq),
        Partition(),
        Map(PCA(), SVMC(), Metric(enhance=False)),
        Summ(function="mean", enhance=False),
        Reduce(),
        Report("mean ... S: $S", enhance=False),
        Report("$X"),
        Report("$y"),
    )
    train, test = pipe.dual_transform()

    print("Train..............\n", train)
    print("Test..........\n", test)


def test_split_train_test(arq="iris.arff"):
    pipe = Pipeline(
        File(arq),
        TrSplit(),
        TsSplit(),
        PCA(),
        SVMC(),
        Metric(enhance=False),
        Report("metric ... R: $R", enhance=False),
    )
    train, test = pipe.dual_transform()

    print("Train..............\n", train)
    print("Test..........\n", test)


def test_with_summ_reduce(arq="iris.arff"):
    pipe = Pipeline(
        File(arq),
        Partition(),
        Map(PCA(), SVMC(), Metric()),
        Map(Report("<---------------------- etapa")),
        Summ(),
        Reduce(),
        Report("mean ... S: $S"),
    )
    train, test = pipe.dual_transform()

    print("Train..............\n", [h.longname for h in train.history])
    print("Test..........\n", [h.longname for h in test.history])


def test_cache(arq="iris.arff"):
    pipe = Pipeline(Cache(File(arq), storage_alias="default_sqlite"))
    train, test = pipe.dual_transform()

    print("Train..............\n", [h.name for h in train.history])
    print("Test..........\n", [h.name for h in test.history])


def test_check_architecture(arq="iris.arff"):
    pipe = Pipeline(
        File(arq),
        Partition(partitions=2),
        Map(PCA(), SVMC(), Metric(enhance=False)),
        Summ(field="Y", function="mean", enhance=False),
    )

    # tenho file na frente
    train_01 = pipe.enhancer.transform(sd.NoData)
    test_01 = pipe.model(sd.NoData).transform(sd.NoData)
    train_02, test_02 = pipe.dual_transform(sd.NoData, sd.NoData)

    # Collection uuid depends on data, which depends on consumption.
    for t, *_ in train_01, train_02, test_01, test_02:
        # print(111111111, t.y)
        pass

    assert train_01.uuid == train_02.uuid
    assert test_01.uuid == test_02.uuid


def test_check_architecture2(arq="iris.arff"):
    pipe = Pipeline(
        File(arq),
        Partition(),
        Map(PCA(), SVMC(), Metric(enhance=False)),
        Summ(field="Y", function="mean", enhance=False),
        Report("mean ... S: $S", enhance=False),
    )

    # tenho file na frente
    train_ = pipe.enhancer.transform(sd.NoData)
    test_ = pipe.model(sd.NoData).transform(sd.NoData)
    test_ = pipe.model(sd.NoData).transform((sd.NoData, sd.NoData))
    train_, test_ = pipe.dual_transform(sd.NoData, sd.NoData)
    train_, test_ = pipe.dual_transform(sd.NoData, (sd.NoData, sd.NoData))

    # train_ = pipe.enhancer().transform()
    # test_ = pipe.model().transform()
    # test_ = pipe.model().transform()
    # train_, test_ = pipe.dual_transform()
    # train_, test_ = pipe.dual_transform()

    # se não tenho file (tenho split)
    # dado = file()
    # train = pipe.enhancer().transform(dado)
    # test = pipe.model(dado).transform(NoData)

    # train = pipe.enhancer().transform(dado)
    # test = pipe.model(dado).transform()

    # se não tenho split
    # dado = file()
    # train, test = split(dado)
    # train = pipe.enhancer().transform(train)
    # test = pipe.model(train).transform(test)

    # train_, test_ = pipe.dual_transform(train, (train, test))

    # chamando info
    # info = pipe.enhancer().info(train)
    # info = pipe.model(train).info()


def printing_test(arq="iris.arff"):
    print(Chain(pja.Map(File(arq))))
    exp = Pipeline(
        File(arq),
        Partition(),
        Map(PCA(), SVMC(), Metric(enhance=False)),
        Map(Report("<---------------------- fold"), enhance=False),
        Summ(function="mean", enhance=False),
        Reduce(),
        Report("mean ... S: $S", enhance=False),
    )
    print(exp)
    print(Select(DT(), SVMC()))

    sel = Select(DT(), SVMC())
    print(sel)
    print(Map(DT()))
    exp = Chain(
        File(arq),
        Partition(),
        pja.Map(PCA(), Select(SVMC(), DT(criterion="gini")), Metric(enhance=False)),
        Report("teste"),
        Map(Report("<---------------------- fold")),
    )
    print(exp)


def random_search(arq="iris.arff"):
    np.random.seed(0)
    exp = Chain(
        File(arq),
        Partition(),
        pja.Map(PCA(), Select(SVMC(), DT(criterion="gini")), Metric()),
        # Map(Report("<---------------------- fold"), enhance=False),
        Summ(function="mean"),
        Reduce(),
        Report("Mean S: $S"),
    )

    # Manual
    a = Sample(exp, n=10)
    result = Best(a, n=5)
    print(len(result.datas))
    print(len(result.components))

    # using Random Search
    rs1 = RandomSearch(exp, sample=20, best=5)
    print(len(rs1.datas))
    print(len(rs1.components))

    rs2 = RandomSearch(rs1, sample=5, best=1)
    print(len(rs2.datas))
    print(len(rs2.components))
    print(rs2.datas[0])
    print(rs2.components[0])

    # Precisamos extrair o pipeline para que ele seja útil para o usuário usar em outras atividade :/
    # Por exemplo, prever outros novos exemplos


def automl_op(arq="iris.arff"):
    np.random.seed(0)
    # AutoML Level
    # Expression --> A * B
    # Workflow --> exp1 * exp2 * ... * expn
    workflow = File(arq) * Partition() * \
               pja.Map(PCA @ MinMax * (SVMC + DT + DT(criterion="gini")) * Metric()) * \
               Summ(function="mean") * \
               Reduce() * Report("Mean S: $S")

    # using Random Search
    rs1 = RandomSearch(workflow, sample=20)
    print(len(rs1.datas))
    print(len(rs1.components))

    res_train, res_test = rs1.datas[0]
    print("Train result: ", res_train)
    print("test result: ", res_test)


def ml_op(arq="iris.arff"):
    np.random.seed(0)
    # ML level --> Pipelines
    pipe = \
        File(arq) * Partition() * \
        Map(PCA() @ MinMax() * (SVMC() + DT(criterion="gini")) * Metric()) * \
        Summ(function="mean") * \
        Reduce() * Report("Mean S: $S")

    print(type(pipe))
    print(pipe)

    res_train, res_test = pipe.dual_transform()
    print("Train result: ", res_train)
    print("test result: ", res_test)


def ger_workflow(seed=0, arq="iris.arff"):
    np.random.seed(seed)

    workflow = Chain(
        File(arq),
        Partition(),
        pja.Map(PCA(), Select(SVMC(), DT(criterion="gini")), Metric(enhance=False)),
        Summ(function="mean", enhance=False),
        Reduce(),
        Report("Mean S: $S", enhance=False),
        seed=seed
    )

    return workflow


def default_config():
    print("SVMC: ", SVMC())

    clist = Sample(SVMC, n=3)
    print(clist)


def avg_cost_of_a_single_sample():
    np.random.seed(0)
    elapsed_times = []
    for i in range(10):
        start_time = withTiming._clock()
        pipes = Sample(ger_workflow(i), n=100)
        delta = round((withTiming._clock() - start_time) * 200) / 10
        elapsed_times.append(delta)
        if i % 3 == 0:
            print(round(i / 0.09), '%   ->', delta, 'ms')
    print("1-sample avg min time: ", min(elapsed_times), "ms")


def test_sequence_of_classifiers(arq="abalone3.arff"):
    pipe = Pipeline(
        File(arq),
        Binarize(), Report('1 {X.shape}'),
        PCA(n=5), SVMC(), Metric(), Report('2 {X.shape}'),
        DT(), Metric(), Report('3 {X.shape} '),
    )
    print('Enh')
    train = pipe.enhancer.transform(sd.NoData)
    print('Mod')
    test = pipe.model(sd.NoData).transform(sd.NoData)  # TODO: pq report não aparece no test?

    print("[test_sequence_of_classifiers] Train.........\n", [h.longname for h in train.history])
    print("[test_sequence_of_classifiers] Test..........\n", [h.longname for h in test.history])


def main():
    """Main function"""
    printable_test()
    test_tsvmc()
    test_split()
    test_metric()
    test_pca()
    test_partition()
    test_split_train_test()
    test_with_summ_reduce()
    printing_test()
    random_search()
    automl_op()
    ml_op()
    default_config()
    avg_cost_of_a_single_sample()
    test_sequence_of_classifiers()

    # test_cache()
    # sanity test
    # test_check_architecture()


if __name__ == "__main__":
    main()

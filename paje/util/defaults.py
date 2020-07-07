from paje.automl.composer.frozen import Frozen
from paje.automl.composer.seq import Seq
from paje.ml.element.modelling.supervised.classifier.dt import DT
from paje.ml.element.modelling.supervised.classifier.knn import KNN
from paje.ml.element.modelling.supervised.classifier.mlp import MLP
from paje.ml.element.modelling.supervised.classifier.nb import NB
from paje.ml.element.modelling.supervised.classifier.rf import RF
from paje.ml.element.modelling.supervised.classifier.svm import SVM
from paje.ml.element.preprocessing.supervised.feature.selector.statistical. \
    chi_square import FilterChiSquare
from paje.ml.element.preprocessing.supervised.feature.selector.statistical. \
    f_score import FilterFScore
from paje.ml.element.preprocessing.supervised.feature.selector.statistical. \
    gini_index import FilterGiniIndex
from paje.ml.element.preprocessing.supervised.feature.selector.statistical. \
    t_score import FilterTScore
from paje.ml.element.preprocessing.supervised.instance.noise_detector. \
    distance_based.nn import NRNN
from paje.ml.element.preprocessing.supervised.instance.sampler.over. \
    ran_over_sampler import RanOverSampler
from paje.ml.element.preprocessing.supervised.instance.sampler.under. \
    ran_under_sampler import RanUnderSampler
from paje.ml.element.preprocessing.unsupervised.feature.scaler.equalization \
    import Equalization
from paje.ml.element.preprocessing.unsupervised.feature.scaler.standard \
    import Standard
from paje.ml.element.preprocessing.unsupervised.feature.transformer.drfa \
    import DRFA
from paje.ml.element.preprocessing.unsupervised.feature.transformer.drgrp \
    import DRGRP
from paje.ml.element.preprocessing.unsupervised.feature.transformer.drpca \
    import DRPCA
from paje.ml.element.preprocessing.unsupervised.feature.transformer.drsrp \
    import DRSRP

# TODO: Extract list of all modules automatically from the root package module?
# TODO: add DRFtAg, DRICA when try/catch is implemented in pipeline execution
ready_classifiers = [DT(), KNN(), MLP(), NB(), RF(), SVM()]
# TODO: AB is not ready and CB is too heavy
ready_transformers = [DRPCA(), DRFA(), DRGRP(), DRPCA(), DRSRP()]
ready_scalers = [Equalization(), Standard()]
pip_chi_squared = [Seq(components=[
    Frozen(Equalization(), feature_range=(0, 1)),
    FilterChiSquare()
])]
# TODO: FilterCFS() broken when called after  DRSRP {'n_components': 15,
#  'density': 0.05, 'dense_output': False, 'eps': 0.7, 'random_state': 0}
#    :  float division by zero, testar com versão anterior do git.
#  já estava quebrando? ficava escondido pelo try/catch?
ready_filters = [FilterFScore(), FilterGiniIndex(),
                 FilterTScore()] + pip_chi_squared
ready_balancing = [RanOverSampler(), RanUnderSampler()]

knn = KNN()
pca = DRPCA()
std = Standard()
eq = Equalization()
pipe_pca = Seq(components=[eq, pca])
pipe2 = Seq(components=[pipe_pca, std, pca])
knn2 = Seq(components=[pipe2, knn])
mlp = Seq(components=[pca, MLP()])

# def_pipelines = [
#     pca,
#     Pipeline(components=[FilterCFS(), mlp])
# ]

default_preprocessors = ready_transformers + ready_scalers + [pca]
default_preprocessors = ready_transformers + ready_scalers + ready_balancing + \
                        ready_filters + [NRNN()]
# default_preprocessors = []
# switch = Switch(components=[Equalization(), Standard()])
# switch2 = Switch(components=[switch, pipe2, pipe_pca])
# default_preprocessors = [pipe_pca, pipe2, switch, switch2]
# default_preprocessors = [pipe_pca, pca, pipe2]

default_modelers = [knn, knn2, mlp] + ready_classifiers
# default_modelers = [knn]

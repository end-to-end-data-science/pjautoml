# from sklearn.externals import joblib
#
# from paje.data.data import Data
# from paje.module.modelling.classifier.rf import RF
# import sklearn.metrics
# from sys import argv
#
# # Some tests to evaluate the resulting size of model dumps.
# from paje.module.preprocessing.supervised.feature.selector.statistical.cfs import FilterCFS
# from paje.pipeline.pipeline import Pipeline
#
# test_size = 1000
#
#
# def f(n):
#     return int(round(n * 1000))
#
#
# if len(argv) != 4:
#     print('Usage: \npython dump.py n_attributes n_classes n_instances')
# else:
#     data = Data.random(int(argv[1]), int(argv[2]), int(argv[3]))
#     x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(data.data_x, data.data_y, random_state=1, test_size=int(int(argv[3]) / 2))
#
#     data_test = Data(x_test, y_test)
#     data_train = Data(x_train, y_train)
#
#     import numpy as np
#     import sklearn
#
#     print('The scikit-learn version is {}.'.format(sklearn.__version__))
#     np.random.seed(1234)
#
#     for x in [100, 100, 1000]:
#         pip = Pipeline([(FilterCFS, {}), (RF, {'n_estimators': x})])
#         pip.show_warnings = False
#         tr = pip.apply(data_train).data_y
#         ts = pip.use(data_test).data_y
#         joblib.dump(pip, pip.name + str(x) + '.dump', compress=('bz2', 9))
#         joblib.dump(pip.obj_comp[1], pip.obj_comp[1].__class__.__name__ + str(x) + '.dump', compress=('bz2', 9))
#
#         print(str(x) + "\t" + str(f(sklearn.metrics.accuracy_score(data_train.data_y, tr))) + "\t" + str(f(sklearn.metrics.accuracy_score(data_test.data_y, ts))))

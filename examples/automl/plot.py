# from paje.data.data import Data
# from paje.module.modelling.classifier.mlp import MLP
# import sklearn.metrics
# from sys import argv
#
# # Some tests to evaluate the behavior of random (no learning, fixed number of neurons) and heuristic (backpropagation, overfitting to number of neurons) adjustment of a neural net (supposing it is analogous to autoML searches).
# # The conclusion is that the probability of overfitting to the validation set is as low as this set is representative of data distribution. Try different values for test_size, for instance.
# test_size = 500
#
#
# def f(n):
#     return int(round(n * 1000))
#
#
# if len(argv) != 3:
#     print('Usage, random search: \npython plot.py dataset.arff rnd')
#     print('Usage, heuristic search: \npython plot.py dataset.arff heu')
# else:
#     data = Data.read_arff(argv[1], "class")
#     x, X_test, y, y_test = sklearn.model_selection.train_test_split(data.data_x, data.data_y, random_state=1, test_size=test_size, train_size=2000)
#     x2, X_test2, y2, y_test2 = sklearn.model_selection.train_test_split(x, y, random_state=1, test_size=test_size)
#     X_train, X_test3, y_train, y_test3 = sklearn.model_selection.train_test_split(x2, y2, random_state=1, test_size=test_size)
#
#     data_test = Data(X_test, y_test)
#     data_test2 = Data(X_test2, y_test2)
#     data_test3 = Data(X_test3, y_test3)
#     data_train = Data(X_train, y_train)
#
#     import numpy as np
#     import sklearn
#
#     print('The scikit-learn version is {}.'.format(sklearn.__version__))
#     np.random.seed(1234)
#
#     for x0 in range(1, 99999):
#         x = int(round((pow(x0, 2)))) if argv[2] == 'heu' else 12
#         model = MLP(activation='relu', hidden_layer_sizes=(x,), max_iter=1, verbose=False, early_stopping=False, validation_fraction=0, batch_size=500, solver='lbfgs') if argv[2] == 'rnd' \
#             else MLP(random_state=1, activation='relu', hidden_layer_sizes=(x,), max_iter=10000, verbose=False, early_stopping=False, validation_fraction=0, batch_size=500, solver='lbfgs')
#         model.show_warns = False
#         tr = model.apply(data_train).data_y
#         ts1 = model.use(data_test).data_y
#         ts2 = model.use(data_test2).data_y
#         ts3 = model.use(data_test3).data_y
#
#         print(str(x) + "\t" + str(f(sklearn.metrics.accuracy_score(data_train.data_y, tr))) + "\t" + str(f(sklearn.metrics.accuracy_score(data_test.data_y, ts1))) + "\t" + str(f(sklearn.metrics.accuracy_score(data_test2.data_y, ts2))) + "\t" + str(f(sklearn.metrics.accuracy_score(data_test3.data_y, ts3))))

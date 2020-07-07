import traceback

msgs = ['All features are either constant or ignored.',  # CB
        'be between 0 and min(n_samples, n_features)',  # DR*
        'excess of max_free_parameters:',  # MLP
        'Timed out!',
        'Mahalanobis for too big data',
        'MemoryError',
        'On entry to DLASCL parameter number',  # Mahala knn
        'excess of neighbors!',  # KNN
        'subcomponent failed',  # nested failure
        'specified nu is infeasible',  # SVM
        'excess of neurons',
        ]


def handle_exception(component, e):
    print('Trying to handle: ' + str(e))
    if not any([str(e).__contains__(msg) for msg in msgs]):
        traceback.print_exc()
        exit(0)
        # raise ExceptionInApplyOrUse(e)


class ExceptionInApplyOrUse(Exception):
    pass


class UseWithoutApply(Exception):
    pass


class ApplyWithoutBuild(Exception):
    pass

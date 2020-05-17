from bayes_opt import BayesianOptimization
from GAN import *

def tune(stock):

    def objective(lr, epoch):
        opt = {}
        opt["lr"] = lr
        opt["epoch"] = int(epoch)
        return -train('GS', opt)

    # Bounded region of parameter space
    pbounds = {'lr': (0.001, 0.005), 'epoch': (100, 150)}

    optimizer = BayesianOptimization(
        f=objective,
        pbounds=pbounds,
        random_state=1,
    )

    optimizer.maximize(
        init_points=2,
        n_iter=3,
    )

    result = optimizer.max["target"]
    return -result


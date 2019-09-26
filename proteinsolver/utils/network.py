from contextlib import contextmanager


@contextmanager
def eval_net(net):
    training = net.training
    try:
        net.train(False)
        yield
    finally:
        net.train(training)

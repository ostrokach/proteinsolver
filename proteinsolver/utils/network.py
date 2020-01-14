from contextlib import contextmanager


@contextmanager
def eval_net(net):
    training = net.training
    try:
        net.eval()
        yield
    finally:
        net.train(training)

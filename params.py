

class mnist:
    batch_size = 50 # 50
    num_worker = 12
    gama = 0.93
    lr = 1e-3
    epoch = 100

    tau = 0.5
    time_step = 4
    num_classes = 10
    alpha = 2
    init_thresh = 2.0
    train_thresh = True
    hete_thresh = 0.01


mnist_para = mnist()


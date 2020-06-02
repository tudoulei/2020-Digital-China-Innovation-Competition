class config():
    def __init__(self):
        self.mode ="train"
        self.use_test = True
        self.use_only_test = False
        self.test_mode = "testA"
        self.train_path = r'./tcdata/hy_round2_train_20200225'
        self.testA_path = r'./tcdata/hy_round2_testA_20200225'
        self.testB_path = None
        self.learning_rate =0.1
        self.epoch = 10



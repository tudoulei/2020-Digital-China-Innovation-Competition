class config():
    def __init__(self):
        # 上线修改这些
        self.online = 1
        self.learning_rate = 0.125
        if self.online:
            self.root_path= "../temp/"
        else:
            self.root_path= "../temp/"    
        ###########################
        self.mode ="train"
        self.use_test = True
        self.use_only_test = False
        self.use_disnear = 0
        self.use_chusai_model = 0
        self.test_mode = "testA"
        self.w2v_save_path= './'
        
        self.train_path = '/tcdata/hy_round2_train_20200225'
        self.testA_path = '/tcdata/hy_round2_testA_20200225'
        self.testB_path = None
        self.use_focalloss =1
        self.epoch = 5000
        
class config_chusai():
    def __init__(self):
        # 上线修改这些
        self.online = 1
        self.learning_rate = 0.1
        self.root_path= "../temp_chusai/"    
        ###########################
        self.mode ="train"
        self.use_test = True
        self.use_only_test = False
        self.use_chusai_model = 0
        self.use_disnear = 0
        self.test_mode = "testA"
        
        self.train_path = '/tcdata/hy_round1_train_20200102'
        self.testA_path = '/tcdata/hy_round2_train_20200225'
        self.testB_path = None
        self.use_focalloss =3 
        self.epoch = 5000
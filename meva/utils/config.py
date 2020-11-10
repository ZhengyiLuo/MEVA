import yaml
import os


class Config:

    def __init__(self, cfg_id, work_dir = ""):
        self.id = cfg_id
        
        cfg_name = os.path.join(work_dir, 'meva/cfg/%s.yml' % cfg_id)
        if not os.path.exists(cfg_name):
            print("Config file doesn't exist: %s" % cfg_name)
            exit(0)
        cfg = yaml.safe_load(open(cfg_name, 'r'))

        self.base_dir = os.path.join(work_dir, "results")
        self.cfg_dir = '%s/meva/%s' % (self.base_dir, cfg_id)
        self.model_dir = '%s/models' % self.cfg_dir
        self.result_dir = '%s/results' % self.cfg_dir
        self.log_dir = '%s/log' % self.cfg_dir
        self.tb_dir = '%s/tb' % self.cfg_dir


        os.makedirs(self.base_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.result_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.tb_dir, exist_ok=True)

        self.batch_size = cfg.get('batch_size', 8)
        self.save_model_interval = cfg.get('save_model_interval', 20)
        self.norm_data = cfg.get('norm_data', True)

        self.lr = cfg['lr']
        self.model_specs = cfg.get('model_specs', dict())
        
        self.num_epoch = cfg['num_epoch']
        self.num_epoch_fix = cfg.get('num_epoch_fix', self.num_epoch)
        self.model_path = os.path.join(self.model_dir, 'model_%04d.p')

        self.data_specs = cfg.get("data_specs", dict())
        self.loss_specs = cfg.get('loss_specs', dict())
        self.num_samples = cfg.get("num_samples", 5000)

if __name__ == '__main__':
    cfg = Config('1212')
    pass


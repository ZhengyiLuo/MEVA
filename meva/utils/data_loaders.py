from dais.utils.dataset_amass_lag import DatasetAMASSLAG
from dais.utils.dataset_amass_lag_debug import DatasetAMASSLAG as DatasetAMASSLAGDEBUG

def get_dataset_cfg(cfg, mode = "all"):
    data_specs = cfg.data_specs
    dataset_name = data_specs['dataset_name']

    if dataset_name == "amass_lag":
        dataset = DatasetAMASSLAG(data_specs, mode = mode)
    elif dataset_name == "amass_lag_debug":
        dataset = DatasetAMASSLAGDEBUG(data_specs, mode = mode)
    else:
        print("unknown dataset!")
        exit()
    return dataset
from meva.dataloaders.dataset_amass import DatasetAMASS

def get_dataset_cfg(cfg, mode = "all"):
    data_specs = cfg.data_specs
    dataset_name = data_specs['dataset_name']

    if dataset_name == "amass_rec":
        dataset = DatasetAMASS(data_specs, mode = mode)
    else:
        print("unknown dataset!")
        exit()
    return dataset
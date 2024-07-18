import os
import json
from simbrain.memristor_fit import MemristorFitting


def full_fitting(device_structure, batch_interval, write_batch_interval):
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
    print(root)
    with open(root + "/memristor_data/sim_params.json") as f:
        sim_params = json.load(f)
    sim_params['device_structure'] = device_structure
    sim_params['batch_interval'] = batch_interval
    sim_params['write_batch_interval'] = write_batch_interval
    
    with open(root + "/memristor_data/my_memristor.json") as f:
        my_memristor = json.load(f)
    print(json.dumps(sim_params, indent=4, separators=(',', ':')))

    exp = MemristorFitting(sim_params, my_memristor)

    if exp.device_name == "mine":
        exp.mem_fitting()
        fitting_record = exp.fitting_record
    else:
        fitting_record = my_memristor

    diff_1 = {k: my_memristor[k] for k in my_memristor if my_memristor[k] != fitting_record[k]}
    diff_2 = {k: fitting_record[k] for k in fitting_record if my_memristor[k] != fitting_record[k]}
    print('Before update:\n', json.dumps(diff_1, indent=4, separators=(',', ':')))
    print('After update:\n', json.dumps(diff_2, indent=4, separators=(',', ':')))

    return sim_params

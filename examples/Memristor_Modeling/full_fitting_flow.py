import os
import json
from simbrain.memristor_fit import MemristorFitting


def full_fitting(device_structure, batch_interval):
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
    print(root)
    with open(root + "/memristor_data/sim_params.json") as f:
        sim_params = json.load(f)
    sim_params['device_structure'] = device_structure
    sim_params['batch_interval'] = batch_interval
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

    # TODO: choose one
    # memristor_device_info.json has been updated in memristor_fit.py, so does fitting_record.json need to be retained?
    # We have:
    # memristor_device_info.json and pkl
    # simbrain/Parameter_files/memristor_device_info.json and pkl
    # memristor_data/fitting_record.json
    with open("../../memristor_data/fitting_record.json", "w") as f:
        json.dump(fitting_record, f, indent=2)

    return sim_params

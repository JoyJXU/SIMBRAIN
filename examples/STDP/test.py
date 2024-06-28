import json

import sys
sys.path.append('../../')

from simbrain.memristor_fit import MemristorFitting


with open("../../memristor_data/sim_params.json") as f:
    sim_params_r = json.load(f)
with open("../../memristor_data/my_memristor.json") as f:
    my_memristor_r = json.load(f)
print(json.dumps(sim_params_r, indent=4, separators=(',', ':')))

exp = MemristorFitting(sim_params_r, my_memristor_r)
if exp.device_name == "mine":
    if exp.mem_size is None:
        print("Error! Missing mem_size.")
    else:
        exp.mem_fitting()
    fitting_record_w = exp.fitting_record
else:
    print("Warning! device_name != mine.")

fitting_record = {k: fitting_record_w[k] for k in fitting_record_w if my_memristor_r[k] != fitting_record_w[k]}
print('\nUpdated parameters:', json.dumps(fitting_record, indent=4, separators=(',', ':')))

with open("../../memristor_data/fitting_record.json", "w") as f:
    json.dump(fitting_record_w, f, indent=2)

import os
import json
import copy
import Fitting_Functions.IV_curve_fitting as IV_curve_fitting
import Fitting_Functions.conductance_fitting as conductance_fitting


class MemristorFitting(object):
    def __init__(
            self,
            sim_params: dict = {},
            my_memristor: dict = {}
    ):
        self.device_name = sim_params['device_name']
        self.c2c_variation = sim_params['c2c_variation']
        self.d2d_variation = sim_params['d2d_variation']
        self.stuck_at_fault = sim_params['stuck_at_fault']
        self.retention_loss = sim_params['retention_loss']
        self.aging_effect = sim_params['aging_effect']
        self.process_nodes = sim_params['process_nodes']
        self.fitting_record = my_memristor

    def mem_fitting(self):
        mem_info = copy.copy(self.fitting_record)

        k_off = mem_info['k_off']
        k_on = mem_info['k_on']
        v_off = mem_info['v_off']
        v_on = mem_info['v_on']
        alpha_off = mem_info['alpha_off']
        alpha_on = mem_info['alpha_on']
        P_off = mem_info['P_off']
        P_on = mem_info['P_on']
        G_off = mem_info['G_off']
        G_on = mem_info['G_on']

        sigma_relative = mem_info['sigma_relative']
        sigma_absolute = mem_info['sigma_absolute']

        Goff_sigma = mem_info['Goff_sigma']
        Gon_sigma = mem_info['Gon_sigma']
        Poff_sigma = mem_info['Poff_sigma']
        Pon_sigma = mem_info['Pon_sigma']

        SAF_lambda = mem_info['SAF_lambda']
        SAF_ratio = mem_info['SAF_ratio']
        SAF_delta = mem_info['SAF_delta']

        retention_loss_tau = mem_info['retention_loss_tau']
        retention_loss_beta = mem_info['retention_loss_beta']

        Aging_k_off = mem_info['Aging_k_off']
        Aging_k_on = mem_info['Aging_k_on']

        # Baseline Model Parameters
        '''
        # Keys don't exist
        if {"k_off", "k_on", "v_off", "v_on", "alpha_off", "alpha_on", "P_off", "P_on", "G_off", "G_on"}.issubset(
        mem_info.keys()):
        '''
        # Values are null
        if None not in [k_off, k_on, v_off, v_on, alpha_off, alpha_on, P_off, P_on, G_off, G_on]:
            pass

        elif os.path.isfile("../memristordata/IV_curve.xlsx") and os.path.isfile("../memristordata/conductance.xlsx"):
            if None in [v_off, v_on]:
                v_off, v_on = 1, -1
                mem_info.update(
                    {
                        "v_off": v_off,
                        "v_on": v_on
                    }
                )
                pass

            if None in [G_off, G_on]:
                G_off, G_on = 1e-6, 1e-8
                mem_info.update(
                    {
                        "G_off": G_off,
                        "G_on": G_on
                    }
                )
                pass

            if None in [alpha_off, alpha_on]:
                alpha_off, alpha_on = IV_curve_fitting.IVCurve(
                    "../memristordata/IV_curve_ferro.xlsx",
                    mem_info
                ).fitting()
                mem_info.update(
                    {
                        "alpha_off": alpha_off,
                        "alpha_on": alpha_on
                    }
                )
                pass

            if None in [P_off, P_on, k_off, k_on]:
                P_off, P_on, k_off, k_on = conductance_fitting.Conductance(
                    "../memristordata/conductance_ferro.xlsx",
                    mem_info
                ).fitting()
                mem_info.update(
                    {
                        "P_off": P_off,
                        "P_on": P_on,
                        "k_off": k_off,
                        "k_on": k_on,
                    }
                )
                pass

            self.fitting_record = mem_info

        else:
            print("Error")

        # Non-ideal Model Parameters
        if self.c2c_variation:
            if None not in [sigma_relative, sigma_absolute]:
                pass
            # elif os.path.isfile(...):
            else:
                # sigma_relative, sigma_absolute = ...
                mem_info.update(
                    {
                        "sigma_relative": sigma_relative,
                        "sigma_absolute": sigma_absolute
                    }
                )
            # else:
            #     print("Error")

        if self.d2d_variation:
            if None not in [Gon_sigma, Goff_sigma, Pon_sigma, Poff_sigma]:
                pass
            # elif...
            else:
                # Goff_sigma, Gon_sigma, Poff_sigma, Pon_sigma = ...
                mem_info.update(
                    {
                        "Goff_sigma": Goff_sigma,
                        "Gon_sigma": Gon_sigma,
                        "Poff_sigma": Pon_sigma,
                        "Pon_sigma": Poff_sigma
                    }
                )
            # else:
            #     print("Error")

        if self.stuck_at_fault:
            if None not in [SAF_lambda, SAF_ratio, SAF_delta]:
                pass
            # elif...
            else:
                # SAF_lambda, SAF_ratio, SAF_delta = ...
                mem_info.update(
                    {
                        "SAF_lambda": SAF_lambda,
                        "SAF_ratio": SAF_ratio,
                        "SAF_delta": SAF_delta
                    }
                )
            # else:
            #     print("Error")

        if self.retention_loss:
            if None not in [retention_loss_tau, retention_loss_beta]:
                pass
            # elif...
            else:
                # retention_loss_tau, retention_loss_beta = ...
                mem_info.update(
                    {
                        "retention_loss_tau": retention_loss_tau,
                        "retention_loss_beta": retention_loss_beta
                    }
                )
            # else:
            #     print("Error")

        if self.aging_effect:
            if None not in [Aging_k_off, Aging_k_on]:
                pass
            # elif...
            else:
                # Aging_k_off, Aging_k_on = ...
                mem_info.update(
                    {
                        "Aging_k_off": Aging_k_off,
                        "Aging_k_on": Aging_k_on
                    }
                )
            # else:
            #     print("Error")

        self.fitting_record = mem_info

        return self.fitting_record


def main():
    # Read dictionary from json
    file = "../memristordata/sim_params.json"
    with open(file) as f:
        sim_params_r = json.load(f)
    file = "../memristordata/my_memristor_ferro.json"
    with open(file) as f:
        my_memristor_r = json.load(f)

    print(json.dumps(sim_params_r, indent=4, separators=(',', ':')))
    # print(json.dumps(my_memristor_r, indent=4, separators=(',', ':')))

    # Run MemristorFitting
    exp = MemristorFitting(sim_params_r, my_memristor_r)
    if exp.device_name == "mine":
        exp.mem_fitting()
        fitting_record_w = exp.fitting_record
    else:
        fitting_record_w = my_memristor_r
    # print(json.dumps(fitting_record_w, indent=4, separators=(',', ':')))
    # Updated parameters
    diff_1 = {k: my_memristor_r[k] for k in my_memristor_r if my_memristor_r[k] != fitting_record_w[k]}
    diff_2 = {k: fitting_record_w[k] for k in fitting_record_w if my_memristor_r[k] != fitting_record_w[k]}
    print('Before update:\n', json.dumps(diff_1, indent=4, separators=(',', ':')))
    print('After update:\n', json.dumps(diff_2, indent=4, separators=(',', ':')))

    # Write dictionary into json
    file = "../memristordata/fitting_record.json"
    with open(file, "w") as f:
        json.dump(fitting_record_w, f, indent=2)

    return 0


if __name__ == '__main__':
    main()

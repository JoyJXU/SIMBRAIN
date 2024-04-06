import os
import json
import copy
import numpy as np
import pandas as pd
import pickle
from simbrain.Fitting_Functions.IV_curve_fitting import IVCurve
from simbrain.Fitting_Functions.conductance_fitting import Conductance
from simbrain.Fitting_Functions.variation_fitting import Variation
from simbrain.Fitting_Functions.retention_loss_fitting import RetentionLoss
from simbrain.Fitting_Functions.aging_effect_fitting import AgingEffect
from simbrain.Fitting_Functions.stuck_at_fault_fitting import StuckAtFault


class MemristorFitting(object):
    """
    Abstract base class for memristor fitting.
    """
    def __init__(
            self,
            sim_params: dict = {},
            my_memristor: dict = {},
            **kwargs,
    ) -> None:
        """
        Abstract base class constructor.
        :param sim_params: Memristor device to be used in learning.
        :param my_memristor: The parameters of the memristor device.
        """

        self.device_name = sim_params['device_name']
        self.c2c_variation = sim_params['c2c_variation']
        self.d2d_variation = sim_params['d2d_variation']
        self.stuck_at_fault = sim_params['stuck_at_fault']
        self.retention_loss = sim_params['retention_loss']
        self.aging_effect = sim_params['aging_effect']
        self.wire_width = sim_params['wire_width']
        self.mem_size = my_memristor['mem_size']
        self.fitting_record = my_memristor

    def mem_fitting(self):
        # %% Obtain memristor parameters
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

        print("Start Memristor Fitting:\n")

        # %% Pre-deployment SAF
        if self.stuck_at_fault in [1, 3]:
            if None not in [SAF_lambda, SAF_ratio]:
                pass
            elif not os.path.isfile(
                    os.path.dirname(os.path.dirname(__file__))
                    + "/memristordata/SAF_data.xlsx"
            ):
                print("Error! Missing data files.\nFailed to update SAF_lambda, SAF_ratio.")
            else:
                print("Pre-deployment Stuck at Fault calculating...")
                SAF_lambda, SAF_ratio = StuckAtFault(
                    os.path.dirname(os.path.dirname(__file__))
                    + "/memristordata/SAF_data.xlsx"
                ).pre_deployment_fitting()
                mem_info.update(
                    {
                        "SAF_lambda": SAF_lambda,
                        "SAF_ratio": SAF_ratio
                    }
                )

        # %%G_off, G_on
        if None not in [G_off, G_on]:
            pass
        elif not os.path.isfile(
                os.path.dirname(os.path.dirname(__file__))
                + "/memristordata/conductance.xlsx"
        ):
            print("Error! Missing data files.\nFailed to update G_off, G_on.")
        else:
            data = pd.DataFrame(pd.read_excel(
                os.path.dirname(os.path.dirname(__file__)) + "/memristordata/conductance.xlsx",
                sheet_name=0,
                header=None,
                names=[
                    'Pulse Voltage(V)',
                    'Current(A)',
                    'Read Voltage(V)'
                ]
            ))
            conductance = np.array(data['Current(A)']) / np.array(data['Read Voltage(V)'][0])
            G_off = np.average(conductance[int(conductance.shape[0] / 2) - 10:int(conductance.shape[0] / 2)])
            G_on = np.average(conductance[conductance.shape[0] - 10:])
            mem_info.update(
                {
                    "G_off": G_off,
                    "G_on": G_on
                }
            )

        # %% D2D variation G_off/G_on
        if self.d2d_variation in [1, 2]:
            if None not in [Gon_sigma, Goff_sigma]:
                pass
            elif None in [G_off, G_on]:
                print("Error! Missing required parameters.\nFailed to update Goff_sigma, Gon_sigma.")
            elif not os.path.isfile(
                    os.path.dirname(os.path.dirname(__file__))
                    + "/memristordata/variation.xlsx"
            ):
                print("Error! Missing data files.\nFailed to update Poff_sigma, Pon_sigma.")
            else:
                print("Device to Device Variation calculating...")
                # TODO: use practical device params instead of mem_info_temp
                mem_info_temp = {
                    'v_off': 2,
                    'v_on': -2,
                    'G_off': 1.11e-9,
                    'G_on': 4.36e-10,
                    'alpha_off': None,
                    'alpha_on': None,
                    'k_off': None,
                    'k_on': None,
                    'P_off': None,
                    'P_on': None,
                    'delta_t': 30 * 1e-3,
                }
                Goff_mu, Goff_sigma, Gon_mu, Gon_sigma = Variation(
                    os.path.dirname(os.path.dirname(__file__))
                    + "/memristordata/variation.xlsx",
                    mem_info_temp
                ).d2d_G_fitting()
                mem_info.update(
                    {
                        "Goff_mu": Goff_mu,
                        "Gon_mu": Gon_mu,
                        "Goff_sigma": Goff_sigma,
                        "Gon_sigma": Gon_sigma,
                    }
                )

        # %% Baseline Model(IV curve)
        # TODO: Use G_mu instead of G in fitting functions if considering D2D Variation
        print("Baseline Model calculating...")
        if None not in [alpha_off, alpha_on]:
            pass
        elif None in [v_off, v_on, G_off, G_on]:
            print("Error! Missing required parameters.\nFailed to update alpha_off, alpha_on.")
        elif not os.path.isfile(
                os.path.dirname(os.path.dirname(__file__))
                + "/memristordata/IV_curve.xlsx"
        ):
            print("Error! Missing data files.\nFailed to update alpha_off, alpha_on.")
        else:
            alpha_off, alpha_on = IVCurve(
                os.path.dirname(os.path.dirname(__file__))
                + "/memristordata/IV_curve.xlsx",
                mem_info
            ).fitting()
            mem_info.update(
                {
                    "alpha_off": alpha_off,
                    "alpha_on": alpha_on
                }
            )

        # %% Baseline Model(Conductance)
        if None not in [P_off, P_on, k_off, k_on]:
            pass
        elif None in [v_off, v_on, G_off, G_on, alpha_off, alpha_on]:
            print("Error! Missing required parameters.\nFailed to update P_off, P_on, k_off, k_on.")
        elif not os.path.isfile(
                os.path.dirname(os.path.dirname(__file__))
                + "/memristordata/conductance.xlsx"
        ):
            print("Error! Missing data files.\nFailed to update P_off, P_on, k_off, k_on.")
        else:
            conductance_temp = (Conductance(
                os.path.dirname(os.path.dirname(__file__))
                + "/memristordata/conductance.xlsx",
                mem_info
            ))
            P_off, P_on, k_off, k_on = conductance_temp.fitting()
            mem_info.update(
                {
                    "P_off": P_off,
                    "P_on": P_on,
                    "k_off": k_off,
                    "k_on": k_on
                }
            )

        # %% D2D variation nonlinearity
        if self.d2d_variation in [1, 3]:
            if None not in [Pon_sigma, Poff_sigma]:
                pass
            elif not os.path.isfile(
                    os.path.dirname(os.path.dirname(__file__))
                    + "/memristordata/variation.xlsx"
            ):
                print("Error! Missing data files.\nFailed to update Poff_sigma, Pon_sigma.")
            else:
                print("Device to Device Variation(Nonlinearity) calculating...")
                # TODO: use practical device params instead of mem_info_temp
                mem_info_temp = {
                    'v_off': 2,
                    'v_on': -2,
                    'G_off': 1.11e-9,
                    'G_on': 4.36e-10,
                    'alpha_off': 5,
                    'alpha_on': 5,
                    'k_off': 20.8,
                    'k_on': -5.2,
                    'P_off': 1.36,
                    'P_on': 0.39,
                    'delta_t': 30 * 1e-3,
                }
                variation_temp = Variation(
                    os.path.dirname(os.path.dirname(__file__))
                    + "/memristordata/variation.xlsx",
                    mem_info_temp
                )
                Poff_sigma, Pon_sigma = variation_temp.d2d_P_fitting()
                mem_info.update(
                    {
                        "Poff_sigma": Pon_sigma,
                        "Pon_sigma": Poff_sigma
                    }
                )

        # %% C2C variation
        if self.c2c_variation:
            if None not in [sigma_relative, sigma_absolute]:
                pass
            elif not os.path.isfile(
                    os.path.dirname(os.path.dirname(__file__))
                    + "/memristordata/variation.xlsx"
            ):
                print("Error! Missing data files.\nFailed to update sigma_relative, sigma_absolute.")
            else:
                print("Cycle to Cycle Variation calculating...")
                if self.d2d_variation not in [1, 3]:
                    sigma_relative, sigma_absolute = conductance_temp.c2c_fitting()
                else:
                    sigma_relative, sigma_absolute = variation_temp.c2c_fitting()
                mem_info.update(
                    {
                        "sigma_relative": sigma_relative,
                        "sigma_absolute": sigma_absolute
                    }
                )

        # %% Post-deployment SAF
        if self.stuck_at_fault in [1, 2]:
            if None not in [SAF_delta]:
                pass
            elif not os.path.isfile(
                    os.path.dirname(os.path.dirname(__file__))
                    + "/memristordata/SAF_data.xlsx"
            ):
                print("Error! Missing data files.\nFailed to update SAF_delta.")
            else:
                print("Post-deployment Stuck at Fault calculating...")
                SAF_delta = StuckAtFault(
                    os.path.dirname(os.path.dirname(__file__))
                    + "/memristordata/SAF_data.xlsx"
                ).post_deployment_fitting()
                mem_info.update(
                    {
                        "SAF_delta": SAF_delta
                    }
                )

        # %% Retention loss
        if self.retention_loss:
            if None not in [retention_loss_tau, retention_loss_beta]:
                pass
            elif not os.path.isfile(
                    os.path.dirname(os.path.dirname(__file__))
                    + "/memristordata/retention_loss.xlsx"
            ):
                print("Error! Missing data files.\nFailed to update retention_loss_tau, retention_loss_beta.")
            else:
                print("Retention Loss calculating...")
                retention_loss_tau, retention_loss_beta = RetentionLoss(
                    os.path.dirname(os.path.dirname(__file__))
                    + "/memristordata/retention_loss.xlsx"
                ).fitting()
                mem_info.update(
                    {
                        "retention_loss_tau": retention_loss_tau,
                        "retention_loss_beta": retention_loss_beta
                    }
                )

        # %% Aging effect
        if self.aging_effect in [1, 2]:
            if None not in [Aging_k_off, Aging_k_on]:
                pass
            elif not os.path.isfile(
                    os.path.dirname(os.path.dirname(__file__))
                    + "/memristordata/aging_effect.xlsx"
            ):
                print("Error! Missing data files.\nFailed to update Aging_k_off, Aging_k_on.")
            else:
                print("Aging Effect calculating...")
                aging_cal = AgingEffect(
                    os.path.dirname(os.path.dirname(__file__))
                    + "/memristordata/aging_effect.xlsx"
                )
                if self.aging_effect == 1:
                    Aging_k_off, Aging_k_on = aging_cal.fitting_equation1()
                else:
                    Aging_k_off, Aging_k_on = aging_cal.fitting_equation2()
                mem_info.update(
                    {
                        "Aging_k_off": Aging_k_off,
                        "Aging_k_on": Aging_k_on
                    }
                )

        print("\nEnd Memristor Fitting.")
        self.fitting_record = mem_info
        
        with open('../../memristor_device_info.json', 'r') as f:
            memristor_info_dict = json.load(f)      
        memristor_info_dict['mine'] = self.fitting_record
        with open('../../memristor_device_info.json', 'w') as f:
            json.dump(memristor_info_dict, f, indent=2)
        
        V_write = np.full(501, 0.975*2*v_off)
        i = 0
        while(V_write[0] > v_off):
            i += 1
            lut_state, lut_conductance = self.memristor_lut_generate(V_write, mem_info)
            if lut_state[50] > 0.975:
                V_write *= 0.975
            elif i == 1:
                for m in range(100,501,50): 
                    if lut_state[m] > 0.975:
                        cut_num = m
                        break
            else:
                break
        if i == 1:
            mine_lut = {        
        'total_no': cut_num,
        'voltage': 0.975*2*v_off,
        'cycle:': 2 * mem_info['delta_t'],
        'duty ratio': 0.5,
        'conductance': lut_conductance[0:cut_num+1]
        }
        else:
            V_write = V_write/0.975
            lut_state, lut_conductance = self.memristor_lut_generate(V_write, mem_info)
            mine_lut = {        
        'total_no': 50,
        'voltage': V_write[0],
        'cycle:': 2 * mem_info['delta_t'],
        'duty ratio': 0.5,
        'conductance': lut_conductance[0:51]
        }
        with open('../../memristor_lut.pkl', 'rb') as f:
            mem_lut = pickle.load(f)                 
        mem_lut['mine'] = mine_lut
        with open('../../memristor_lut.pkl', 'wb') as f:
            pickle.dump(mem_lut, f)
        
        return self.fitting_record

    def memristor_lut_generate(self, V_write, mem_info):
        J1 = 1
        points = len(V_write)

        # initialization
        internal_state = [0 for i in range(points)]
        conductance_fit = [0 for i in range(points)]

        # conductance change
        internal_state[0] = 0
        for i in range(points - 1):
            if V_write[i + 1] > mem_info['v_off'] and V_write[i + 1] > 0:
                delta_x = mem_info['k_off'] * ((V_write[i + 1] / mem_info['v_off'] - 1) ** mem_info['alpha_off']) * J1 * (
                        (1 - internal_state[i]) ** mem_info['P_off'])
                internal_state[i + 1] = internal_state[i] + mem_info['delta_t'] * delta_x
            elif V_write[i + 1] < mem_info['v_on'] and V_write[i + 1] < 0:
                delta_x = mem_info['k_on'] * ((V_write[i + 1] / mem_info['v_on'] - 1) ** mem_info['alpha_on']) * J1 * (
                        internal_state[i] ** mem_info['P_on'])
                internal_state[i + 1] = internal_state[i] + mem_info['delta_t'] * delta_x                     
            else:
                delta_x = 0
                internal_state[i + 1] = internal_state[i]
            if internal_state[i + 1] < 0: 
                internal_state[i + 1] = 0
            elif internal_state[i + 1] > 1:
                internal_state[i + 1] = 1

        # conductance calculation
        for i in range(points):
            conductance_fit[i] = mem_info['G_off'] * internal_state[i] + mem_info['G_on'] * (1 - internal_state[i])

        return internal_state, conductance_fit

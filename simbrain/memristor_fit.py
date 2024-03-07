import os
import copy
from simbrain.Fitting_Functions.IV_curve_fitting import IVCurve
from simbrain.Fitting_Functions.conductance_fitting import Conductance


class MemristorFitting(object):
    """
    Abstract base class for fitting the parameters of the memristor device.
    """

    def __init__(
            self,
            sim_params: dict = {},
            my_memristor: dict = {},
            **kwargs,
    ):
        """
        Abstract base class constructor.
        :param sim_params: Memristor device to be used in learning
        :param my_memristor: The parameters of the memristor device.
        """
        super().__init__()

        self.device_name = sim_params['device_name']
        self.c2c_variation = sim_params['c2c_variation']
        self.d2d_variation = sim_params['d2d_variation']
        self.stuck_at_fault = sim_params['stuck_at_fault']
        self.retention_loss = sim_params['retention_loss']
        self.aging_effect = sim_params['aging_effect']
        self.process_nodes = sim_params['process_nodes']
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

        # %% Pre-deployment SAF
        if self.stuck_at_fault in [1, 3]:
            if None not in [SAF_lambda, SAF_ratio]:
                pass
            # elif None in []:
            # elif not os.path.isfile(...):
            else:
                # SAF_lambda, SAF_ratio = ...
                mem_info.update(
                    {
                        "SAF_lambda": SAF_lambda,
                        "SAF_ratio": SAF_ratio
                    }
                )

        # %%G_off, G_on
        if None not in [G_off, G_on]:
            pass
        # elif None in []:
        # elif not os.path.isfile(...):
        else:
            # G_off, G_on = ...
            mem_info.update(
                {
                    "G_off": G_off,
                    "G_on": G_on
                }
            )

        # %% D2D variation G_off/G_on
        if self.d2d_variation in [1, 3]:
            if None not in [Gon_sigma, Goff_sigma]:
                pass
            # elif None in []:
            # elif not os.path.isfile(...):
            else:
                # Goff_sigma, Gon_sigma = ...
                mem_info.update(
                    {
                        "Goff_sigma": Goff_sigma,
                        "Gon_sigma": Gon_sigma,
                    }
                )

        # %% Baseline Model(IV curve)
        if None not in [alpha_off, alpha_on]:
            pass
        elif None in [v_off, v_on, G_off, G_on]:
            print("Error! Missing required parameters.\nFailed to update alpha_off, alpha_on.")
        elif not os.path.isfile(os.path.dirname(os.path.dirname(__file__)) + "/memristordata/IV_curve.xlsx"):
            print("Error! Missing data files.\nFailed to update alpha_off, alpha_on.")
        else:
            alpha_off, alpha_on = IVCurve(
                os.path.dirname(os.path.dirname(__file__)) + "/memristordata/IV_curve.xlsx",
                mem_info
            ).fitting()
            mem_info.update(
                {
                    "alpha_off": alpha_off,
                    "alpha_on": alpha_on
                }
            )

        # %% Baseline Model(IV curve)
        if None not in [P_off, P_on, k_off, k_on]:
            pass
        elif None in [v_off, v_on, G_off, G_on, alpha_off, alpha_on]:
            print("Error! Missing required parameters.\nFailed to update P_off, P_on, k_off, k_on.")
        elif not os.path.isfile(os.path.dirname(os.path.dirname(__file__)) + "/memristordata/conductance.xlsx"):
            print("Error! Missing data files.\nFailed to update P_off, P_on, k_off, k_on.")
        else:
            P_off, P_on, k_off, k_on = Conductance(
                os.path.dirname(os.path.dirname(__file__)) + "/memristordata/conductance.xlsx",
                mem_info
            ).fitting()
            mem_info.update(
                {
                    "P_off": P_off,
                    "P_on": P_on,
                    "k_off": k_off,
                    "k_on": k_on
                }
            )

        # %% C2C variation
        if self.c2c_variation:
            if None not in [sigma_relative, sigma_absolute]:
                pass
            # elif None in []:
            # elif not os.path.isfile(...):
            else:
                # sigma_relative, sigma_absolute = ...
                mem_info.update(
                    {
                        "sigma_relative": sigma_relative,
                        "sigma_absolute": sigma_absolute
                    }
                )

        # %% D2D variation nonlinearity
        if self.d2d_variation in [1, 2]:
            if None not in [Pon_sigma, Poff_sigma]:
                pass
            # elif None in []:
            # elif not os.path.isfile(...):
            else:
                # Poff_sigma, Pon_sigma = ...
                mem_info.update(
                    {
                        "Poff_sigma": Pon_sigma,
                        "Pon_sigma": Poff_sigma
                    }
                )

        # %% Post-deployment SAF
        if self.stuck_at_fault in [1, 2]:
            if None not in [SAF_delta]:
                pass
            # elif None in []:
            # elif not os.path.isfile(...):
            else:
                # SAF_delta = ...
                mem_info.update(
                    {
                        "SAF_delta": SAF_delta
                    }
                )

        # %% Retention loss
        if self.retention_loss in [1, 2]:
            if None not in [retention_loss_tau, retention_loss_beta]:
                pass
            # elif None in []:
            # elif not os.path.isfile(...):
            else:
                # retention_loss_tau, retention_loss_beta = ...
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
            # elif None in []:
            # elif not os.path.isfile(...):
            else:
                # Aging_k_off, Aging_k_on = ...
                mem_info.update(
                    {
                        "Aging_k_off": Aging_k_off,
                        "Aging_k_on": Aging_k_on
                    }
                )

        self.fitting_record = mem_info

        return self.fitting_record

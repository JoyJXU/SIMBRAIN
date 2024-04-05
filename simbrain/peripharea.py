import torch
from typing import Iterable, Optional, Union
from simbrain.formula import Formula
import math
import json


class PeriphArea(torch.nn.Module):
    def __init__(
            self,
            sim_params: dict = {},
            shape: Optional[Iterable[int]] = None,
            CMOS_tech_info_dict: dict = {},
            memristor_info_dict: dict = {},
            **kwargs,
    ) -> None:
        # language=rst
        """
        Abstract base class constructor.

        :param sim_params: Memristor device to be used in learning.
        :param shape: The dimensionality of the crossbar.
        :param memristor_info_dict: The parameters of the memristor device.
        :param length_row: The physical length of the horizontal wire in the crossbar.
        :param length_col: The physical length of the vertical wire in the crossbar.
        """
        super().__init__()

        self.shape = shape
        self.device_name = sim_params['device_name']
        self.device_structure = sim_params['device_structure']
        self.device_roadmap = sim_params['device_roadmap']
        self.ADC_precision = sim_params['ADC_precision']
        self.input_bit = sim_params['input_bit']
        self.temperature = sim_params['temperature']
        self.CMOS_technode = sim_params['CMOS_technode']
        self.CMOS_technode_meter = sim_params['CMOS_technode'] * 1e-9
        self.CMOS_tech_info_dict = CMOS_tech_info_dict
        self.memristor_info_dict = memristor_info_dict

        self.MIN_NMOS_SIZE = self.CMOS_tech_info_dict['Constant']['MIN_NMOS_SIZE']
        self.MAX_TRANSISTOR_HEIGHT = self.CMOS_tech_info_dict['Constant']['MAX_TRANSISTOR_HEIGHT']
        self.IR_DROP_TOLERANCE = self.CMOS_tech_info_dict['Constant']['IR_DROP_TOLERANCE']
        self.POLY_WIDTH = self.CMOS_tech_info_dict['Constant']['POLY_WIDTH']
        self.MIN_GAP_BET_GATE_POLY = self.CMOS_tech_info_dict['Constant']['MIN_GAP_BET_GATE_POLY']
        self.pnSizeRatio = self.CMOS_tech_info_dict[self.device_roadmap][str(self.CMOS_technode)]['pn_size_ratio']
        self.Goff = self.memristor_info_dict[self.device_name]['G_off']

        # self.widthInvN = self.MIN_NMOS_SIZE * self.CMOS_technode_meter
        # self.widthInvP = self.pnSizeRatio * self.MIN_NMOS_SIZE * self.CMOS_technode_meter

        self.formula_function = Formula(sim_params=sim_params, shape=self.shape,
                                        CMOS_tech_info_dict=self.CMOS_tech_info_dict)


    def Adder_area_calculation(self, newHeight, newWidth):
        widthNandN = 2 * self.MIN_NMOS_SIZE * self.CMOS_technode_meter
        widthNandP = self.pnSizeRatio * self.MIN_NMOS_SIZE * self.CMOS_technode_meter
        wNand, hNand = self.formula_function.calculate_gate_area("NAND", 2, widthNandN, widthNandP,
                                                                 self.CMOS_technode_meter * self.MAX_TRANSISTOR_HEIGHT)
        numAdder = self.shape[1]
        if newHeight:
            # Adder in multiple columns given the total height
            self.hAdder = hNand
            self.wAdder = wNand * 9 * self.ADC_precision

            # Calculate the number of adder per column
            numAdderPerCol = int(newHeight / self.hAdder)
            if numAdderPerCol > numAdder:
                numAdderPerCol = numAdder
            numColAdder = int(math.ceil(numAdder / numAdderPerCol))
            self.adder_height = newHeight
            self.adder_width = self.wAdder * numColAdder

        elif newWidth:
            # Adder in multiple rows given the total width
            self.hAdder = hNand * self.ADC_precision
            self.wAdder = wNand * 9

            # Calculate the number of adder per row
            numAdderPerRow = int(newWidth / self.wAdder)
            if numAdderPerRow > numAdder:
                numAdderPerRow = numAdder
            numRowAdder = int(math.ceil(numAdder / numAdderPerRow))
            self.adder_width = newWidth
            self.adder_height = self.hAdder * numRowAdder

        else:
            # Assume one row of adder by default
            self.hAdder = hNand
            self.wAdder = wNand * 9 * self.ADC_precision
            self.adder_width = self.wAdder * numAdder
            self.adder_height = self.hAdder

        self.adder_area = self.adder_height * self.adder_width

        return self.adder_area


    def ShiftAdder_area_calculation(self, newHeight, newWidth):
        numDff = (self.ADC_precision + self.input_bit) * self.shape[1]
        if newWidth:
            self.Adder_area_calculation(None, newWidth)
            self.DFF_area_calculation(None, newWidth, numDff)

        else:
            print("[ShiftAdd] Error: No width assigned for the shift-and-add circuit")
            exit(-1)

        self.ShiftAdder_height = self.adder_height + self.CMOS_technode_meter * self.MAX_TRANSISTOR_HEIGHT + self.Dff_height
        self.ShiftAdder_width = newWidth

        self.ShiftAdder_area = self.ShiftAdder_height * self.ShiftAdder_width

        return self.ShiftAdder_area


    def DFF_area_calculation(self, newHeight, newWidth, numDff):
        # Assume DFF size is 12 minimum-size standard cells put together
        # widthTgN = self.MIN_NMOS_SIZE * self.CMOS_technode_meter
        # widthTgP = self.pnSizeRatio * self.MIN_NMOS_SIZE * self.CMOS_technode_meter
        wDffInv, hDffInv = self.formula_function.calculate_gate_area("INV", 1,
                                                                     self.MIN_NMOS_SIZE * self.CMOS_technode_meter,
                                                                     self.pnSizeRatio * self.MIN_NMOS_SIZE * self.CMOS_technode_meter,
                                                                     self.CMOS_technode_meter * self.MAX_TRANSISTOR_HEIGHT)
        hDff = hDffInv
        wDff = wDffInv * 12

        if newHeight:  # DFF in multiple columns given the total height
            # Calculate the number of DFF per column
            numDFFPerCol = int(newHeight / hDff)
            if numDFFPerCol > numDff:
                numDFFPerCol = numDff
            numColDFF = int(math.ceil(numDff / numDFFPerCol))
            self.Dff_height = newHeight
            self.Dff_width = wDff * numColDFF

        elif newWidth:  # DFF in multiple rows given the total width
            # Calculate the number of DFF per row
            numDFFPerRow = int(newWidth / wDff)
            if numDFFPerRow > numDff:
                numDFFPerRow = numDff
            numRowDFF = int(math.ceil(numDff / numDFFPerRow))
            self.Dff_width = newWidth
            self.Dff_height = hDff * numRowDFF

        else:  # Assume one row of DFF by default
            self.Dff_width = wDff * numDff
            self.Dff_height = hDff

        self.Dff_area = self.Dff_height * self.Dff_width

        return self.Dff_area


    def SarADC_area_calculation(self, heightArray, widthArray):
        widthNmos = self.MIN_NMOS_SIZE * self.CMOS_technode_meter
        widthPmos = self.pnSizeRatio * self.MIN_NMOS_SIZE * self.CMOS_technode_meter
        wNmos, hNmos = self.formula_function.calculate_gate_area("INV", 1, widthNmos, 0,
                                                                 self.CMOS_technode_meter * self.MAX_TRANSISTOR_HEIGHT)
        wPmos, hPmos = self.formula_function.calculate_gate_area("INV", 1, 0, widthPmos,
                                                                 self.CMOS_technode_meter * self.MAX_TRANSISTOR_HEIGHT)
        levelOutput = 2 ^ self.ADC_precision

        self.areaUnit = (hNmos * wNmos) * (269 + (math.log2(levelOutput) - 1) * 109) + (hPmos * wPmos) * (
                    209 + (math.log2(levelOutput) - 1) * 73)

        area = 0
        if widthArray:
            self.SarADC_area = self.areaUnit * self.shape[1]
            self.width = widthArray
            self.height = area / widthArray
        elif heightArray:
            self.SarADC_area = self.areaUnit * self.shape[1]
            self.height = heightArray
            self.width = area / heightArray
        else:
            print("[MultilevelSenseAmp] Error: No width or height assigned for the multiSenseAmp circuit")
            exit(-1)

            # Assume the Current Mirrors are on the same row and the total width of them is smaller than the adder or DFF

        return self.SarADC_area


    def Switchmatrix_area_calculation(self, newHeight, newWidth, mode):
        resMemCellOnAtVw = 1 / self.Goff
        if mode == "ROW_MODE":  # Connect to rows
            minCellHeight = self.MAX_TRANSISTOR_HEIGHT * self.CMOS_technode_meter
            resTg = resMemCellOnAtVw / self.shape[1] * self.IR_DROP_TOLERANCE
            widthTgN = self.formula_function.calculate_on_resistance(self.CMOS_technode_meter,
                                                                     "NMOS") * self.CMOS_technode_meter / (resTg * 2)
            widthTgP = self.formula_function.calculate_on_resistance(self.CMOS_technode_meter,
                                                                     "PMOS") * self.CMOS_technode_meter / (resTg * 2)
            if newHeight:
                if newHeight < minCellHeight:
                    print("[SwitchMatrix] Error: pass gate height is even larger than the array height")
                numTgPairPerCol = int(
                    newHeight / minCellHeight)  # Get max # Tg pair per column (this is not the final # Tg pair per column because the last column may have less # Tg)
                numColTgPair = int(math.ceil(
                    self.shape[0] / numTgPairPerCol))  # Get min # columns based on this max # Tg pair per column
                numTgPairPerCol = int(
                    math.ceil(self.shape[0] / numColTgPair))  # Get # Tg pair per column based on this min # columns
                TgHeight = newHeight / numTgPairPerCol
                wTg, hTg = self.formula_function.calculate_gate_area("INV", 1, widthTgN, widthTgP, TgHeight)

                # DFF
                numDff = self.shape[0]
                self.DFF_area_calculation(newHeight, None, numDff)

                self.Switchmatrix_height = newHeight
                self.Switchmatrix_width = (wTg * 2) * numColTgPair + self.Dff_width

            else:
                wTg, hTg = self.formula_function.calculate_gate_area("INV", 1, widthTgN, widthTgP,
                                                                     minCellHeight)  # Pass gate with folding
                self.Switchmatrix_height = hTg * self.shape[0]
                numDff = self.shape[0]
                self.DFF_area_calculation(self.Switchmatrix_height, None,
                                          numDff)  # Need to give the height information, otherwise by default the area calculation of DFF is in column mode
                self.Switchmatrix_width = (wTg * 2) + self.Dff_width

        else:  # Connect to columns
            resTg = resMemCellOnAtVw / self.shape[0] / 2
            widthTgN = self.formula_function.calculate_on_resistance(self.CMOS_technode_meter,
                                                                     "NMOS") * self.CMOS_technode_meter / (resTg * 2)
            widthTgP = self.formula_function.calculate_on_resistance(self.CMOS_technode_meter,
                                                                     "PMOS") * self.CMOS_technode_meter / (resTg * 2)
            if newWidth:
                minCellWidth = 2 * (
                            self.POLY_WIDTH + self.MIN_GAP_BET_GATE_POLY) * self.CMOS_technode_meter  # min standard cell width for 1 Tg
                if minCellWidth > newWidth:
                    print("[SwitchMatrix] Error: pass gate width is even larger than the array width")
                numTgPairPerRow = int(newWidth / (
                            minCellWidth * 2))  # Get max # Tg pair per row (this is not the final # Tg pair per row because the last row may have less # Tg)
                numRowTgPair = int(
                    math.ceil(self.shape[1] / numTgPairPerRow))  # Get min # rows based on this max # Tg pair per row
                numTgPairPerRow = int(
                    math.ceil(self.shape[1] / numRowTgPair))  # Get # Tg pair per row based on this min # rows
                TgWidth = newWidth / numTgPairPerRow / 2  # division of 2 because there are 2 Tg in one pair
                numFold = int(TgWidth / (0.5 * minCellWidth)) - 1  # get the max number of folding

                # widthTgN, widthTgP and numFold can determine the height and width of each pass gate
                wTg, hTg = self.formula_function.calculate_pass_gate_area(widthTgN, widthTgP, numFold)

                # DFF
                numDff = self.shape[0]
                self.DFF_area_calculation(None, newWidth, numDff)

                self.Switchmatrix_width = newWidth
                self.Switchmatrix_height = hTg * numRowTgPair + self.Dff_height

            else:
                # Default (pass gate with folding=1)
                wTg, hTg = self.formula_function.calculate_pass_gate_area(widthTgN, widthTgP, 1)
                self.Switchmatrix_width = wTg * 2 * self.shape[0]
                numDff = self.shape[0]
                self.DFF_area_calculation(None, self.Switchmatrix_width, numDff)
                self.Switchmatrix_height = hTg + self.Dff_height
        self.Switchmatrix_area = self.Switchmatrix_height * self.Switchmatrix_width

        return self.Switchmatrix_area



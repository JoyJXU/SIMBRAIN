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
        self.numBit = sim_params['ADC_accuracy']
        self.inputbit = sim_params['input_bit']
        self.temperature = sim_params['temperature']
        self.CMOS_technode = int(sim_params['CMOS_technode']) * 1e-9
        self.CMOS_technode_inNano = int(sim_params['CMOS_technode'])
        self.CMOS_technode_str = sim_params['CMOS_technode']

        with open('../../CMOS_tech_info.json', 'r') as file:
            self.CMOS_tech_info_dict = json.load(file)
        self.MIN_NMOS_SIZE = self.CMOS_tech_info_dict['Constant']['MIN_NMOS_SIZE']
        self.MAX_TRANSISTOR_HEIGHT = self.CMOS_tech_info_dict['Constant']['MAX_TRANSISTOR_HEIGHT']
        self.IR_DROP_TOLERANCE = self.CMOS_tech_info_dict['Constant']['IR_DROP_TOLERANCE']
        self.POLY_WIDTH = self.CMOS_tech_info_dict['Constant']['POLY_WIDTH']
        self.MIN_GAP_BET_GATE_POLY = self.CMOS_tech_info_dict['Constant']['MIN_GAP_BET_GATE_POLY']
        self.pnSizeRatio = self.CMOS_tech_info_dict[self.device_roadmap][self.CMOS_technode_str]['pn_size_ratio']

        with open('../../memristor_device_info.json', 'r') as f:
            self.memristor_info_dict = json.load(f)
        assert self.device_name in self.memristor_info_dict.keys(), "Invalid Memristor Device!"
        self.Goff = self.memristor_info_dict[self.device_name]['G_off']

        self.widthTgN = self.MIN_NMOS_SIZE * self.CMOS_technode
        self.widthTgP = self.pnSizeRatio * self.MIN_NMOS_SIZE * self.CMOS_technode
        self.widthInvN = self.MIN_NMOS_SIZE * self.CMOS_technode
        self.widthInvP = self.pnSizeRatio * self.MIN_NMOS_SIZE * self.CMOS_technode
        self.widthNandN = 2 * self.MIN_NMOS_SIZE * self.CMOS_technode
        self.widthNandP = self.pnSizeRatio * self.MIN_NMOS_SIZE * self.CMOS_technode

        self.formula_ = Formula(sim_params=sim_params, shape=self.shape)

    def Adder_area_calculation(self, newHeight, newWidth):
        gate_area_result_adder = self.formula_.calculate_gate_area("NAND", 2, self.widthNandN, self.widthNandP, self.CMOS_technode * self.MAX_TRANSISTOR_HEIGHT)
        hNand = gate_area_result_adder[0]
        wNand = gate_area_result_adder[1]
        numAdder = self.shape[1]
        if newHeight:
            # Adder in multiple columns given the total height
            self.hAdder = hNand
            self.wAdder = wNand * 9 * self.numBit

            # Calculate the number of adder per column
            numAdderPerCol = int(newHeight / self.hAdder)
            if numAdderPerCol > numAdder:
                numAdderPerCol = numAdder
            numColAdder = int(math.ceil(numAdder / numAdderPerCol))
            self.adder_height = newHeight
            self.adder_width = self.wAdder * numColAdder

        elif newWidth:
            # Adder in multiple rows given the total width
            self.hAdder = hNand * self.numBit
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
            self.wAdder = wNand * 9 * self.numBit
            self.adder_width = self.wAdder * numAdder
            self.adder_height = self.hAdder

        self.adder_area = self.adder_height * self.adder_width

        return self.adder_area

    def ShiftAdder_area_calculation(self, newHeight, newWidth):
        numDff = (self.numBit + self.inputbit) * self.shape[1]
        if newWidth:
            self.Adder_area_calculation(None, newWidth)
            self.DFF_area_calculation(None, newWidth, numDff)
            
        else:
            print("[ShiftAdd] Error: No width assigned for the shift-and-add circuit")
            exit(-1)

        self.ShiftAdder_height = self.adder_height + self.CMOS_technode * self.MAX_TRANSISTOR_HEIGHT + self.Dff_height
        self.ShiftAdder_width = newWidth

        self.ShiftAdder_area = self.ShiftAdder_height * self.ShiftAdder_width

        return self.ShiftAdder_area
        
    def DFF_area_calculation(self, newHeight, newWidth, numDff):
        # Assume DFF size is 12 minimum-size standard cells put together
        gate_area_result_Dff = self.formula_.calculate_gate_area("INV", 1, self.MIN_NMOS_SIZE * self.CMOS_technode, self.pnSizeRatio * self.MIN_NMOS_SIZE * self.CMOS_technode, self.CMOS_technode * self.MAX_TRANSISTOR_HEIGHT)
        hDffInv = gate_area_result_Dff[0]
        wDffInv = gate_area_result_Dff[1]
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
        widthNmos = self.MIN_NMOS_SIZE * self.CMOS_technode
        widthPmos = self.pnSizeRatio * self.MIN_NMOS_SIZE * self.CMOS_technode
        gate_area_result_SarADC_N = self.formula_.calculate_gate_area("INV", 1, widthNmos, 0, self.CMOS_technode * self.MAX_TRANSISTOR_HEIGHT)
        gate_area_result_SarADC_P = self.formula_.calculate_gate_area("INV", 1, 0, widthPmos, self.CMOS_technode * self.MAX_TRANSISTOR_HEIGHT)
        hNmos = gate_area_result_SarADC_N[0]
        wNmos = gate_area_result_SarADC_N[1]
        hPmos = gate_area_result_SarADC_P[0]
        wPmos = gate_area_result_SarADC_P[1]
        levelOutput = 2 ^ self.numBit

        self.areaUnit = (hNmos * wNmos) * (269 + (math.log2(levelOutput) - 1) * 109) + (hPmos * wPmos) * (209 + (math.log2(levelOutput) - 1) * 73)

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
        resTg = resMemCellOnAtVw / self.shape[1] * self.IR_DROP_TOLERANCE
        self.widthTgN = self.formula_.calculate_on_resistance(self.CMOS_technode, "NMOS") * self.CMOS_technode / (resTg * 2) 
        self.widthTgP = self.formula_.calculate_on_resistance(self.CMOS_technode, "PMOS") * self.CMOS_technode / (resTg * 2)
        if mode == "ROW_MODE":  # Connect to rows
            minCellHeight = self.MAX_TRANSISTOR_HEIGHT * self.CMOS_technode

            if newHeight:
                if newHeight < minCellHeight:
                    print("[SwitchMatrix] Error: pass gate height is even larger than the array height")
                numTgPairPerCol = int(newHeight / minCellHeight)  # Get max # Tg pair per column (this is not the final # Tg pair per column because the last column may have less # Tg)
                self.numColTgPair = int(math.ceil(self.shape[0] / numTgPairPerCol))  # Get min # columns based on this max # Tg pair per column
                numTgPairPerCol = int(math.ceil(self.shape[0] / self.numColTgPair))  # Get # Tg pair per column based on this min # columns
                self.TgHeight = newHeight / numTgPairPerCol
                gate_area_result_Switchmatrix = self.formula_.calculate_gate_area("INV", 1, self.widthTgN, self.widthTgP, self.TgHeight)
                hTg = gate_area_result_Switchmatrix[0]
                wTg = gate_area_result_Switchmatrix[1]

                # DFF
                numDff = self.shape[0]
                self.DFF_area_calculation(newHeight, None, numDff)

                self.Switchmatrix_height = newHeight
                self.Switchmatrix_width = (wTg * 2) * self.numColTgPair + self.Dff_width

            else:
                gate_area_result_Switchmatrix = self.formula_.calculate_gate_area("INV", 1, self.widthTgN, self.widthTgP, minCellHeight)  # Pass gate with folding
                hTg = gate_area_result_Switchmatrix[0]
                wTg = gate_area_result_Switchmatrix[1]
                self.Switchmatrix_height = hTg * self.shape[0]
                numDff = self.shape[0]
                self.DFF_area_calculation(self.Switchmatrix_height, None, numDff)  # Need to give the height information, otherwise by default the area calculation of DFF is in column mode
                self.Switchmatrix_width = (wTg * 2) + self.Dff_width

        else:  # Connect to columns
            if newWidth:
                minCellWidth = 2 * (self.POLY_WIDTH + self.MIN_GAP_BET_GATE_POLY) * self.CMOS_technode  # min standard cell width for 1 Tg
                if minCellWidth > newWidth:
                    print("[SwitchMatrix] Error: pass gate width is even larger than the array width")
                numTgPairPerRow = int(newWidth / (minCellWidth * 2))  # Get max # Tg pair per row (this is not the final # Tg pair per row because the last row may have less # Tg)
                self.numRowTgPair = int(math.ceil(self.shape[0] / numTgPairPerRow))  # Get min # rows based on this max # Tg pair per row
                numTgPairPerRow = int(math.ceil(self.shape[0] / self.numRowTgPair))  # Get # Tg pair per row based on this min # rows
                self.TgWidth = newWidth / numTgPairPerRow / 2  # division of 2 because there are 2 Tg in one pair
                numFold = int(self.TgWidth / (0.5 * minCellWidth)) - 1  # get the max number of folding

                # widthTgN, widthTgP and numFold can determine the height and width of each pass gate
                gate_area_result_Switchmatrix = self.formula_.calculate_pass_gate_area(self.widthTgN, self.widthTgP, numFold)
                hTg = gate_area_result_Switchmatrix[0]
                wtg = gate_area_result_Switchmatrix[1]

                # DFF
                numDff = self.shape[0]
                self.DFF_area_calculation(None, newWidth, numDff)

                self.width = newWidth
                self.height = hTg * self.numRowTgPair + self.Dff_height

            else:
                # Default (pass gate with folding=1)
                gate_area_result_Switchmatrix = self.formula_.calculate_pass_gate_area(self.widthTgN, self.widthTgP, 1)
                hTg = gate_area_result_Switchmatrix[0]
                wTg = gate_area_result_Switchmatrix[1]
                self.Switchmatrix_width = wTg * 2 * self.shape[0]
                numDff = self.shape[0]
                self.DFF_area_calculation(None, self.Switchmatrix_width, numDff)
                self.Switchmatrix_height = hTg + self.Dff_height
        self.Switchmatrix_area = self.Switchmatrix_height * self.Switchmatrix_width

        capTgGateN = self.formula_.calculate_gate_cap(self.widthTgN)
        capTgGateP = self.formula_.calculate_gate_cap(self.widthTgP)

        return self.Switchmatrix_area




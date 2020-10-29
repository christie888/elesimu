import numpy as np
import math
import pandas as pd


class Elevator(object):
    #Static paramters of the building

    def __init__(self,Floor_Num):
        self.T_m= 5 # moving time between two floors
        self.T_b_base = 5  # boarding time base + (*)*passenger count transfered at the floor
        self.N =Floor_Num  # total floor number
        self.C = 15  # capacity of cabin
        self.Ele_num=math.ceil(self.N/5)  #Elevator Numbers depending on the floor numbers
        self.Ele_Capa= self.C*self.Ele_num

        self.in_up_traffic = np.arange(self.N)
        self.in_down_traffic = np.arange(self.N)
        self.off_traffic = np.arange(self.N)

class Building(object):
    #Static paramters of the building

    def __init__(self, Floor_Num):
        self.TL= np.zeros(Floor_Num)
        self.TL[1:] =200
        self.Pc =np.sum(self.TL)


class Result(object):
    #Static paramters of the building

    def __init__(self,Floor_Num):
        self.WaitingTimeNext = 0;
        self.llsum = 0
        self.lltotalsum = 0
        self.transferred_Count = 0;


        #self.round_num =0

        #self.transferredMt = np.zeros(Floor_Num);  # transffered traffic load matrix

        #self.transferred_tl_df = pd.DataFrame()
        #self.remained_pc_df = pd.DataFrame()
        self.origin_tl_df = pd.DataFrame()
        self.merge_tl_df = pd.DataFrame()
        self.tt_df = pd.DataFrame()
        self.ll_sum_df = pd.DataFrame()
        self.targetingF_df = pd.DataFrame()

        #self.transferredMt = np.zeros(Floor_Num)
        #self.transferredCM_df = pd.DataFrame()
        self.over_pc_array = list()
        self.changeNum_array = list()
        self.wtsum_array = list()
        self.wtNext_array = list()
        self.ttsum_array = list()
        #self.wt_for_next_array = list()
        #self.pc_array = list()
        self.round_num_array = list()
        self.llsum_array = list()
        self.duTime_array = list()
        self.PCsum = np.array(self.origin_tl_df).sum()

    def getPCsum(self):
        print("transferred pc sum ---",np.array(self.origin_tl_df).sum())
        return np.array(self.origin_tl_df).sum()


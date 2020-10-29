from elesimu.ElevatorEnv import *
from elesimu.Functions import *
import os


MuFilePath ='file/mu/mu_test.xlsx'

Floor_Num = 20
Mu_flag = 0
pattern ="normal" #"mdp"#"merge"# #"normal" #"normal" #

Dire_FLg= "down"  #"down" #
Delta_f= 2

Peak_duration = 2 * 3600
Delta_T = 5 * 60
time_intervals = Peak_duration / Delta_T

duTime = 0
round_num=1
mdpChangeNum=0
#result_sheet_name = str(pattern) + str(Dire_FLg) + "detF"+str(Delta_f) +"detime"+ str(Delta_T)

def ManageTL(tl,changeTemp,mdpChangeNum,mu_dist):
    tl_managed=0
    ll=0
    targetFs=np.zeros(Floor_Num);

    if pattern=="normal":
        tl_managed= tl
    elif pattern=="merge":
        targetFs = initargetFs
        mergeresult = Merge(initargetFs, tl, Delta_f, Floor_Num)
        tl_managed = mergeresult[0]
        ll = mergeresult[1]
    else:
        if mdpChangeNum==0 or changeTemp>mdpChangeNum:
            print(changeTemp)
            if Mu_flag == 0:
                fileNormalpath = './file/normalData/' + paras + str(Mu_flag) + '_normal.xlsx'
                normaldata = pd.read_excel(fileNormalpath)

                targetFs = tagetingFloorMDP(Delta_f, duTime, normaldata, Delta_T, Floor_Num)
            else:
                print("mdp change time====", mdpChangeNum)
                targetFs = tagetingFloorMDPMu(Delta_f, mdpChangeNum, mu_dist, Floor_Num)
            mdpChangeNum += 1
            # duTime_array.append(duTime)

        mergeResult = MergeMDP(targetFs, tl, Delta_f, Mu_flag,Floor_Num)  # return new tl, labor loss distribution
        tl_managed = mergeResult[0]
        ll = mergeResult[1]

    return tl_managed,ll,targetFs


if __name__ == "__main__":

    ele = Elevator(Floor_Num)
    bld =Building(Floor_Num)
    rst=Result(Floor_Num)

    WaitingTimeNext=0

    Pc = bld.Pc

    P_a =  Peak_duration/Pc;

    roundNum=0

    paras = "FN_" + str(Floor_Num) + "_Time_" + str(Peak_duration) + "pc_" + str(int(Pc))

    sheetName= str(Dire_FLg)+"_"+str(pattern)+"_Mu_" + str(Mu_flag)
    print(paras,sheetName)
    createResultFile(paras,sheetName)

    mu_dist=0

    if Mu_flag == 1:  # peak overlap
        if pattern == "normal":
            peakMap = peak_map(Floor_Num)
            #print(peakMap)
            mu_df = CreateMuDistribution(Floor_Num, peakMap)
            drawkkde(mu_df) #draw the kde graph
            mu_dist = mu_array(mu_df, Floor_Num)

            #print(mu_dist)

            writeIntoMuDistribution(mu_dist, MuFilePath)
        else:
            if os.path.exists(MuFilePath):
                mu_dist = pd.read_excel(MuFilePath)
            else:
                print("Please Generate Mu file")


    initargetFs = iniateTarFs(Delta_f, Floor_Num)
    maxTripTime=0

    while True:
        changeTemp = duTime // Delta_T + 1

        #print(duTime, changeTemp)
        roundNum += 1
        #print(roundNum)

        # create traffic load
        if Mu_flag == 1:
            mu=mu_dist[changeTemp]
            Tl_rst= GenerateTL_Mu(P_a,mu,Floor_Num, WaitingTimeNext, ele, bld, rst)
            #GenerateTL_Random(P_a, Floor_Num, WaitingTimeNext, ele, bld, rst)
        else:
            Tl_rst= GenerateTL_Random(P_a, Floor_Num, WaitingTimeNext, ele, bld,rst)

        wt_sum =Tl_rst[0]
        tem_tl =Tl_rst[1]
        overPC =Tl_rst[2]

        bld.TL-=tem_tl

        tl_managed_result= ManageTL(tem_tl,changeTemp,mdpChangeNum, mu_dist)

        #print(tl_managed_result[0])
        #print(tl_managed_result[1])
        mergeTl=tl_managed_result[0]
        labor_effort = tl_managed_result[1]
        targetFs =tl_managed_result[2]

        #maxTripTime= lambda x: tl_managed_result[3] if tl_managed_result[3]> maxTripTime else maxTripTime

        trans_result=TransportTL(mergeTl,Dire_FLg,WaitingTimeNext, Floor_Num,ele)
        #print(trans_result)

        WaitingTimeNext = trans_result[0]
        tt_sum=trans_result[1]
        longtt=trans_result[3]/ele.Ele_num

        #print(tt_sum)

        if Dire_FLg=="down":
            wt=trans_result[2]
            wt_sum = mergeTl*wt

        if overPC==0:
            wt_sum +=overPC*longtt

        print("long trip time ",longtt)

        duTime += longtt
        RecordTL(wt_sum, tem_tl, overPC, rst, bld, ele, tl_managed_result,trans_result, WaitingTimeNext,roundNum,changeTemp,targetFs, duTime)

        print(np.sum(rst.origin_tl_df))
        #if roundNum>=10:
            #break;

        #if rst.PCsum >= Pc:
        if rst.getPCsum() >= 1900:
            break;

    processResult(rst,duTime,roundNum,paras,sheetName)






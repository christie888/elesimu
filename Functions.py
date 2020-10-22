import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
import math
import numpy as np


def CreateMuDistribution(Floor_Num,peakMap):
    n=25
    rs = np.random.RandomState(1979)
    x = np.random.normal(2, 2, size=int(n*(Floor_Num-1))) # mean, std, count
    g=np.tile(np.arange(1,Floor_Num), n) # repeat by 25
    df = pd.DataFrame(dict(x=x, g=g))
    map = df.g.map(peakMap)
    df["x"] += map

    return df


def peak_map(Floor_Num):
    dic = {}
    for i in range(Floor_Num-1):
        posi = np.random.randint(2, Floor_Num-1)  # peak time position
        dic[i+1] =posi

    return dic


def mu_array(df, N):
    N = 20  # 楼层
    n =25  # 变化次数
    mu_array = np.zeros(int(N * n)).reshape(N, int(n))

    #mu_array+=0.1
    for i in range(N):
        sub_df=df.loc[df['g'] == i]

        muarr=mapPosi(sub_df,n)

        mu_array[i]=np.random.normal(0.1, 0.05, n)+muarr


    mu_array[0] = 0
    return mu_array


def mapPosi(sub_df,n):
    muarr = np.zeros(n)
    for each in sub_df["x"]:

        re=math.modf(float(each))
        posi = re[1]
        prob = float(re[0])

        if prob >= 0.5:
            prob -= 0.3

        if(int(posi)< n):
            muarr[int(posi)]=float('%.2f' % prob)
    return muarr


def drawkkde(df):
    sns.set(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
    pal = sns.cubehelix_palette(20, rot=-.25, light=.7)
    g = sns.FacetGrid(df, row="g", hue="g", aspect=15, size=.5, palette=pal)  # height=.5,

    # Draw the densities in a few steps
    g.map(sns.kdeplot, "x", clip_on=False, shade=True, alpha=1, lw=1.5, bw=.2)
    g.map(sns.kdeplot, "x", clip_on=False, color="w", lw=2, bw=.2)
    g.map(plt.axhline, y=0, lw=2, clip_on=False)

    # Define and use a simple function to label the plot in axes coordinates
    def label(x, color, label):
        ax = plt.gca()

        ax.text(-0.05, .2, "Floor "+label, fontweight="bold", color=color,
                ha="left", va="center", transform=ax.transAxes, fontsize=14)

        ax.tick_params(axis='both', which='major', labelsize=16)

    g.map(label, "x")

    # Set the subplots to overlap
    g.fig.subplots_adjust(hspace=-.25)

    # Remove axes details that don't play well with overlap
    g.set_titles("")
    g.set(yticks=[])
    g.despine(bottom=True, left=True)

    plt.xlim(0, 25)

    plt.savefig("mu.png")

    plt.show()

def writeIntoMuDistribution(mu_dist, fileNormalpath):
    # fileNormalpath = './file/normalData/mu_distribution' + paras + '.xlsx'
    writer = pd.ExcelWriter(fileNormalpath)
    pd.DataFrame(mu_dist).to_excel(writer, 'mu distribution')
    writer.save()


def destination_floor_generate(tl,Floor_num, building_TL):
    while True:
        tem_f=np.random.randint(1, Floor_num)
        if building_TL[tem_f]>0:
            building_TL[tem_f]-=1
            tl[tem_f] += 1
            break;
    return tl

def GenerateTL_Random(P_ab, Floor_num, WaitingTimeNext, ele, bld,result):
    building_TL =bld.TL
    PCsum = result.getPCsum()
    pc = 0;
    over_pc = 0
    temWTSum = 0
    Pc = bld.Pc

    #WaitingTimeNext=120

    if (WaitingTimeNext == 0):
        # print("waiting timme ==0: ----------------")
        tl = np.random.randint(5, size=Floor_num)
        tl[0] = 0
        PCsum += np.sum(tl)
        building_TL = building_TL - tl
        temWTSum = np.sum(tl) * (P_ab + 1)
    else:
        # print("waiting timme =" +str(WaitingTimeNextMerge)+"----------------")
        tem_time = 0;
        tl = np.zeros(Floor_num)
        while True:
            mu_inter = np.random.normal(P_ab, 0.5)  # generate a overall arrival rate mu by normal distribution : mean std; s/pc
            #tp = mu_inter
            if pc < ele.Ele_Capa:
                tl = destination_floor_generate(tl,Floor_num, building_TL)  # generate the target floor number of the person
                pc += 1
                PCsum += 1
            else:
                over_pc += 1

            tem_time += mu_inter
            temWTSum += (WaitingTimeNext - tem_time)  # add waiting time for a single

            if tem_time >= WaitingTimeNext:
                break;
            if PCsum >= Pc:
                break;

    #print(pc+over_pc)
    #print(over_pc)

    return [temWTSum, tl, over_pc]  # 0:总等待时间； 1:交通量；3：过量的人数

def GenerateTL_Mu(P_ab, mu,Floor_Num, WaitingTimeNext, ele, bld, result):
    building_TL = bld.TL
    PCsum = result.getPCsum()
    pc = 0;
    over_pc = 0
    temWTSum = 0
    Pc = bld.Pc

    #WaitingTimeNext = 120

    if (WaitingTimeNext == 0):
        # print("waiting timme ==0: ----------------")
        tl = np.random.randint(5, size=Floor_Num)
        tl[0] = 0
        PCsum += np.sum(tl)
        building_TL = building_TL - tl
        temWTSum = np.sum(tl) * (P_ab + 1)
    else:
        tem_time = 0;
        while True:
            mu_inter = np.random.normal(P_ab, 0.5)  # generate a overall arrival rate mu by normal distribution : mean std; s/pc
            #tp = mu_inter

            if pc < ele.Ele_Capa:
                pc += 1
                PCsum += 1
            else:
                over_pc += 1
            tem_time += mu_inter
            temWTSum += (WaitingTimeNext - tem_time)  # add waiting time for a single

            if tem_time >= WaitingTimeNext:
                break;
            #print(Pc)
            #print(PCsum)

            if PCsum >= Pc:
                break;

        tl = destination_floor_generate_withMu(pc,Floor_Num, mu)


    return [temWTSum, tl, over_pc]  # 0:总等待时间； 1:交通量；3：过量的人数

def destination_floor_generate_withMu(pc,Floor_num,mu):
    tl = np.zeros(Floor_num)
    totalmu=np.sum(mu)

    for i in range(Floor_num):
        p=mu[i]/totalmu
        tl[i]=np.ceil(p*pc)
    return tl


def RecordTL(wt_sum,tem_tl,overPC,rst,bld, ele, merge_tl, trans_result,WaitingTimeNext,roundNum, changeTemp,targetFs,duTime):
    rst.wtsum_array.append(wt_sum)
    rst.over_pc_array.append(overPC)

    rst.origin_tl_df=pd.concat([rst.origin_tl_df,pd.DataFrame(tem_tl)], axis=1)
    rst.merge_tl_df = pd.concat([rst.merge_tl_df, pd.DataFrame(merge_tl[0])], axis=1)
    rst.targetingF_df = pd.concat([rst.targetingF_df ,pd.DataFrame(targetFs)], axis=1)

    ll_sum=tem_tl*merge_tl[1]
    rst.ll_sum_df= pd.concat([rst.ll_sum_df, pd.DataFrame(ll_sum)], axis=1)

    tt_sum=round(np.sum(trans_result[1]*merge_tl[0]), 2)

    rst.wtNext_array.append(WaitingTimeNext)
    #print(tt_sum)
    rst.ttsum_array.append(tt_sum)
    rst.round_num_array.append(roundNum)
    rst.changeNum_array.append(changeTemp)
    rst.duTime_array.append(duTime)


    #bld.Pc-= np.sum(rst.origin_tl_df)

    #rst.wtsum_array
def iniateTarFs(delF_Merge,N):
    targetF=np.zeros(N)
    i=N-1
    while i-delF_Merge>=0:
        targetF[i-delF_Merge]=1
        i-=delF_Merge+1
    return targetF


def Merge(targetFs, origin, f_Merge,Floor_Num):
    #tem = temTL
    tl = np.zeros(Floor_Num)
    ll=np.zeros(Floor_Num)

   # for i in range(N):
    i=0
    while True:
        tl[i] = origin[i]
        if(targetFs[i]==1):
            #tl[i] = origin[i]
            j=1
            while True:
                tl[i]+=origin[i+j]
                ll[i+j] =j
                j+=1
                if j>f_Merge:
                    i += f_Merge
                    break
        i+=1
        if i >=Floor_Num:
            break;

    return tl,ll

def TransportTL(temTL,dire,pre_wt, N, ele):
    trip_time = 0
    wt = np.zeros(N)
    tt = np.zeros(N)

    T_b=ele.T_m
    T_m=ele.T_b_base
    Ele_num = ele.Ele_num


    if dire == "up":
        floor_start = 0
        f_c = floor_start

        while True:
            if temTL[f_c] != 0:  # if temTL[1] != 0:
                tt[f_c] = trip_time
                boardingTime = T_b + temTL[f_c] * 0.2  # single person prelong 0.2 s
                trip_time += boardingTime
                # tt[f_c, 1] = int(wt[f_c, 1]) + int(wt[f_c, 2])
            f_c += 1;
            trip_time += T_m

            if (f_c == N):
                #longtt = trip_time + (N - 1) * T_m
                longtt = np.amax(tt) + (N - 1) * T_m
                WaitingTimeNext = longtt/ Ele_num;
                break;
    else:
        floor_start = N - 1
        f_c = floor_start
        print("pre waiting", pre_wt)
        waitime = pre_wt
        # waitime=(N-1) * T_m
        tem_cometime = np.zeros(N)

        while True:
            waitime += T_m
            if temTL[f_c] != 0:
                # waitime = pre_wt
                boardingTime = T_b + temTL[f_c] * 0.2
                # print("pick up at: "+str(f_c)+" , people count  " +str(trafficLoad[f_c, 2]))
                trip_time += boardingTime;
                waitime += boardingTime
                wt[f_c] = waitime  # waiting
                # +int(T_b) trip time
                # wt[f_c, 3] = wt[f_c, 1] + wt[f_c, 2]
            # print("current waiting time " + str(wt[f_c, 1]) + " trip time:  " +str(wt[f_c, 2]) + "  journal time : "+str(wt[f_c, 3]))

            trip_time += T_m
            waitime += T_m
            tem_cometime[f_c] = trip_time
            # waitime += T_m
            f_c -= 1;

            if f_c < 0:
                temptt = tt + trip_time
                tt = temptt - tem_cometime;
                longtt = np.amax(tt) + (N - 1) * T_m
                wt = wt / Ele_num
                WaitingTimeNext = np.amax(wt)

                break;

    return WaitingTimeNext, tt, wt, longtt


def MergeMDP(targetFs, origin, delF_Merge,flag,N):
    tl = np.zeros(N)
    i = N-1
    ll = np.zeros(N)
    targIndexArray =np.where(targetFs==1)[0]
    while True:

        startLambda=lambda i: i - delF_Merge if i - delF_Merge > 0 else 0
        startIndex=startLambda(i)
        endIndex=i+1
        sliceUnit=origin[startIndex:endIndex]

        startIndexs= np.where(targIndexArray >= startIndex, True, False)
        endIndexs = np.where(targIndexArray < endIndex, True, False)
        combin= (startIndexs ==endIndexs)
        wher=np.where(combin == True)[0]

        if wher:
            targIndUnit = targIndexArray[wher][0]

            if targetFs[i]==1:
                tl[i] = np.sum(sliceUnit) #traffic load sum in the unit
                j=1
                while True:
                    temIndex = i - j
                    if targetFs[temIndex]==0:
                        ll[temIndex] = abs(temIndex - targIndUnit)
                    j+=1
                    if j>=delF_Merge+1:
                        break
            else:
                ll[i]=abs(i-targIndUnit)
                j=1
                while True:
                    temIndex = i - j
                    if targetFs[temIndex] == 1:
                        tl[temIndex] = np.sum(sliceUnit)
                    else:
                        ll[temIndex] = abs(temIndex- targIndUnit)
                    j += 1
                    if j >=delF_Merge + 1:
                        break

        i -= delF_Merge + 1
        if i < 0:
            break;

    #print(tl)

    return tl, ll


def tagetingFloorMDP(delF_Merge, duTime,normaldata,deltaT,N):
    #print(normaldata)
    timeindex=np.array(normaldata.columns)

    targetFlist = np.zeros(N)
    startTime=duTime
    endTime= startTime+deltaT
    cols=list()

    print(startTime, endTime)

    for timestamp in timeindex:
        if timestamp>=startTime and timestamp<endTime:
            cols.append(timestamp)

    print(cols)
    print(normaldata[cols])
    predicSum=np.sum(normaldata[cols],axis=1)

    print(predicSum)
    i=N-1
    while True:
        sliceEnd=i+1;
        #slcieStart=i-delF_Merge

        slcieStartlambda = lambda i: i - delF_Merge if i - delF_Merge>0 else 0
        slcieStart=slcieStartlambda (i)
        #print(" slcieStart,sliceEnd", slcieStart , sliceEnd)
        tem= pd.Series(predicSum[slcieStart:sliceEnd])
        #print("temp sum",tem)
        #tarIndex=np.argmax(tem)
        tarIndex = pd.Series.idxmax(tem)
        targetFlist[tarIndex]=1 #mark the target floor with 1
        #print("target index", tarIndex)
        i -= delF_Merge + 1

        if i-delF_Merge+1<0:
            break;
    #filter= timeindex>=startTime and timeindex<endTime
    #print(filter)

    #timeindex[startTime < timeindex < endTime]=1111

    return targetFlist


def tagetingFloorMDPMu(delF_Merge, mdpChangeNum,mu_dist, N):
    print(mu_dist, mdpChangeNum)
    print(mu_dist[mdpChangeNum])
    curren_mu=mu_dist[mdpChangeNum]

    targetFlist = np.zeros(N)

    i = N - 1
    while True:
        sliceEnd = i + 1;
        # slcieStart=i-delF_Merge

        slcieStartlambda = lambda i: i - delF_Merge if i - delF_Merge > 0 else 0
        slcieStart = slcieStartlambda(i)
        # print(" slcieStart,sliceEnd", slcieStart , sliceEnd)
        tem = curren_mu[slcieStart:sliceEnd]

        print(slcieStart,sliceEnd)
        # print("temp sum",tem)
        # tarIndex=np.argmax(tem)
        tarIndex = pd.Series.idxmax(tem)
        targetFlist[tarIndex] = 1  # mark the target floor with 1
        # print("target index", tarIndex)
        i -= delF_Merge + 1

        if i - delF_Merge + 1 < 0:
            break;

    print(targetFlist)
    return targetFlist
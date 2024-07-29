import os
import pandas as pd
import numpy as np
import xml.dom.minidom
import xml.etree.ElementTree as ET
def dtype_xml(filepath):
    files = os.listdir(filepath)
    os.chdir(filepath)
    for filename in files:
            portion = os.path.splitext(filename)
            try:
                if portion[1] == ".rsml":
                   tmp_name = portion[0] + ".xml"
                   os.rename(filename, tmp_name)
            except:
                print(filename)
    print("Convert finish")

def listname(filepath,savepath):
# filepath = "F:/Root/Total/Root_trait(Total)/Root measurements.csv"
#  filepath = "F:/Root/Total/Root_trait(Total)/Plant measurements.csv"
#  rootdata = pd.read_csv(filepath, header=None)
# 设置error_bad_lines=False 跳过报错行
 rootdata = pd.read_csv(filepath,error_bad_lines=False)
# 获取数据信息
 rootdata.info()
# 行标签
# .colums 列标签
 root_Tag = rootdata['Tag'].astype(str)
 Tmp_rootdata = []
 Tag = []
 Tag1 = []
# Root measurements 表格Tag处理
#  for index in range(len(root_Tag)):
#   Tmp_tag = root_Tag.iloc[index]
#   judge1 = Tmp_tag.find('.')
#   judge2 = Tmp_tag.find(':')
#   if judge1 < judge2:
#    Tag.append(Tmp_tag[0:judge1])
#   else:
#    Tag.append(Tmp_tag[0:judge2])
#  tmp = pd.DataFrame(data=Tag)
# # tmp.to_csv('tmp.csv')
#  tmp.to_csv(savepath,encoding='gbk')
#  print('Done')
#Plant measurements 表格处理

 for index in range(len(root_Tag)):
   # 指定行
   Tmp_tag = root_Tag.iloc[index]
   # judge1 = Tmp_tag.find('.')
   judge2 = Tmp_tag.find(':')
   # if judge1 > 0:
   #   Tag.append(Tmp_tag[0:judge1])
   # else:
   Tag.append(Tmp_tag[0:judge2])
 tmp = pd.DataFrame(data=Tag)
 # tmp.to_csv('tmp.csv')
 tmp.to_csv(savepath,encoding='gbk')
 return tmp
 print('Done')

def cal_LPcount(Data,Tag,Labeled):
        Tag = Data[Tag]
        Labeled = Data[Labeled]
        LPcount = []
        tmp_PrimaryNum = 0
        tmp_num = 1
        AverageLPCount = []
        MaxLPCount = []
        MinLPCount = []
        tmp_LPcount = []
        tmp_index = []
        num = 0
        i = 0
        l_num = 0
        for index_labeled in range(len(Labeled)):
         tmp_labeled = Labeled[index_labeled]
         index_1 = index_labeled + 1
         try:
            tmp_labeled_1 = Labeled.iloc[index_1]
            if(tmp_labeled == tmp_labeled_1):
                tmp_num += 1
            else:
                for index_Tag in range(i, tmp_num):
                  tmp_tag = Tag[index_Tag]
                  judge = tmp_tag.find(':')
                  judge += 5
                  try:
                     if (tmp_tag[judge - 3] != '.'):
                         judge += 1
                     tmp_tag[judge]
                     l_num += 1
                     i = tmp_tag[judge]
                     tmp_LPcount.append(l_num)

                  except:
                      # tmp_LPcount_P = 0
                      tmp_PrimaryNum += 1
                      l_num = 0
                if tmp_LPcount == []:
                    LPcount = [0,0]
                else:
                    Num = tmp_LPcount
                    i_1 = 0
                    for m in range(len(tmp_LPcount)):
                        tmp_LPcountnum = tmp_LPcount[m]
                        m_1 = m + 1
                        try:
                            tmp_LPcountnum_1 = tmp_LPcount[m_1]
                            if tmp_LPcountnum >= tmp_LPcountnum_1:
                                LPcount.append(tmp_LPcount[m])
                        except:
                            LPcount.append(tmp_LPcount[m])
                            #     tmp_index.append(tmp_LPcountnum)
                            # else:
                            #     tmp_index.append(tmp_LPcountnum)
                            #     try:
                            #         tmp_LPcount[m_1 + 1]
                            #     except:
                            #         tmp_index.append(tmp_LPcount[m_1])
                            #
                        #         LPcount.append(max(tmp_index[i_1:m_1]))
                        #         i_1 = m_1
                        # except:
                        #         if len(tmp_index[i_1:len(tmp_LPcount)]) > 1:
                        #            tmp_index.append(tmp_LPcount[m])
                        #         LPcount.append(tmp_index[m])

                        # except:



                #   LPcount.append(tmp_LPcount_P)
                # item = [m for m, x in enumerate(LPcount) if x == 0]
                # if len(item) == len(LPcount):
                #  for m in range(len(LPcount)):
                #      Num.append(LPcount[m])
                # else:
                #    for m in range(len(LPcount)):
                #       if LPcount[m] != 0:
                #           num += 1
                #       else:
                #          Num.append(num)
                #          num = 0
                #          for index_value in range(len(Num) - 1, -1, -1):
                #              if (Num[index_value] == 0):
                #                  Num.remove(0)
                #          continue
                LPcount_item = []
                if LPcount != []:
                    for item in range(len(LPcount)):
                        LPcount_item.append(int(LPcount[item]))
                    tmp_AverageLPCount = sum(LPcount_item) / tmp_PrimaryNum
                else:
                    tmp_AverageLPCount = 0
                # tmp_AverageLPCount = sum(Num) / (len(Num) - Num.count(0))
                tmp_MaxLPCount = max(LPcount)
                tmp_MinLPCount = min(LPcount)
                tmp_PrimaryNum = 0
                i = tmp_num
                tmp_num += 1
                LPcount = []
                tmp_LPcount = []
                tmp_index = []
                Num = []
                AverageLPCount.append(tmp_AverageLPCount)
                MaxLPCount.append(tmp_MaxLPCount)
                MinLPCount.append(tmp_MinLPCount)

         except:
            print("cal_LPcount invalid")
            print(Tag[index_Tag + 1])
            print(".......")
        return  AverageLPCount, MaxLPCount, MinLPCount

def cal_PrimaryTrait(Data,Tag,Labeled,Trait):
 # filename = "F:/Root/Total/Root_trait(Total)/Root measurements._1.csv"
 # Data = pd.read_csv(filename)
 try:
   Average_value = []
   Tol_value = []
   Max_value = []
   Min_value =[]
   value = []
   Num = []
   Tag = Data[Tag]
   Labeled = Data[Labeled]
   Trait = Data[Trait]
   tmp_num = 1
   tmp_num1 = 0
   i = 0
   for index_labeled in range(len(Labeled)):
     tmp_labeled = Labeled[index_labeled]
     index_1 = index_labeled + 1
     try:
      tmp_labeled_1 = Labeled.iloc[index_1]
      if (tmp_labeled == tmp_labeled_1):
        tmp_num += 1
      else:
        for index_Tag in range(i, tmp_num):
          tmp_tag = Tag[index_Tag]
          judge = tmp_tag.find(':')
          judge += 5
          try:
            tmp_tag[judge]
          except:
            tmp_num1 += 1
            tmp_value = abs(Trait[index_Tag])
            value.append(tmp_value)
        Average_tmp_value = np.mean(value)
        Tol_tmp_value = sum(value)
        Max_tmp_value = max(value)
        Min_tmp_value = min(value)
        Average_value.append(Average_tmp_value)
        Tol_value.append(Tol_tmp_value)
        Max_value.append(Max_tmp_value)
        Min_value.append(Min_tmp_value)
        value = []
        i = tmp_num
        tmp_num1 = 0
        tmp_num += 1
     except:
       print("cal_PrimaryTrait")
       print(tmp_labeled)
       print("invalid")
      # Num.append(tmp_num1)

     continue
 except:
   print("cal_PrimaryTrait")
   print(Tag[index_Tag + 1])
   print("..........")
 return Tol_value, Average_value, Max_value, Min_value

def cal_LataralTrait(Data,Tag,Labeled,Trait):
# filename = "G:/Root/Total/Root_trait(Total)/Root measurements._1.csv"
#  Data = pd.read_csv(filename)
 try:
   Average_value = []
   Tol_value = []
   Max_value = []
   Min_value = []
   value = []
   Num = []
   Tag = Data[Tag]
   Labeled = Data[Labeled]
   Trait = Data[Trait]
   tmp_num = 1
   tmp_num1 = 0
   i = 0
   for index_labeled in range(len(Labeled)):
     tmp_labeled = Labeled[index_labeled]
     index_1 = index_labeled + 1
     try:
      tmp_labeled_1 = Labeled.iloc[index_1]
      if (tmp_labeled == tmp_labeled_1):
        tmp_num += 1
      else:
        for index_Tag in range(i, tmp_num):
          tmp_tag = Tag[index_Tag]
          judge = tmp_tag.find(':')
          judge += 5
          try:
            tmp_tag[judge]
            tmp_value = abs(Trait[index_Tag])
            # value.append(tmp_value)
          except:
            tmp_num1 += 1
            tmp_value = 'a'
          value.append(tmp_value)
          try:
            Tol_tmp_value = sum(value)
          except:
              # Tol_tmp_value == 0
              # if Tol_tmp_value == 0:
              #     Max_tmp_value = 0
              #     Min_tmp_value = 0
              # else:
                  for index_value in range(len(value)-1,-1,-1):
                      if (value[index_value] == "a") :
                          value.remove("a")
        if value == []:
            value = [0]
            Tol_tmp_value = 0
        Tol_tmp_value = np.sum(value)
        Average_tmp_value = np.mean(value)
        Max_tmp_value = max(value)
        Min_tmp_value = min(value)
        Average_value.append(Average_tmp_value)
        Tol_value.append(Tol_tmp_value)
        Max_value.append(Max_tmp_value)
        Min_value.append(Min_tmp_value)
        value = []
        i = tmp_num
        tmp_num1 = 0
        tmp_num += 1
     except:
       print("cal_LataralTrait")
       print(tmp_labeled)

        # Num.append(tmp_num1)

     continue
 except:
   print("cal_LataralTrait")
   print(Tag[index_Tag + 1])
   print("..........")
 return Tol_value, Average_value, Max_value, Min_value

  # return Average_value
def cal_LataralCount(Data, num1, num2):
  valid_num = len(Data[num1:num2]) - list(Data[num1:num2]).count(0)
  return valid_num
def cal_PrimaryCount(Data,Tag,labeled):
# filename = "G:/Root/Total/Root_trait(Total)/Root measurements._1.csv"
# Data = pd.read_csv(filename)
 try:
  Num = []
  Tag = Data[Tag]
  Labeled = Data[labeled]
  tmp_num = 1
  tmp_num1 = 0
  i = 0
  for index_labeled in range(len(Labeled)):
    tmp_labeled = Labeled[index_labeled]
    index_1 = index_labeled + 1
    try:
     tmp_labeled_1 = Labeled.iloc[index_1]
     if (tmp_labeled == tmp_labeled_1):
       tmp_num += 1
     else:
       for index_Tag in range(i,tmp_num ):
         # print(tmp_num)
         tmp_tag = Tag[index_Tag]
         judge = tmp_tag.find(':')
         judge += 5
         try:
           tmp_tag[judge]
         except:
           tmp_num1 += 1
       Num.append(tmp_num1)
       i = tmp_num
       tmp_num1 = 0
       tmp_num += 1
    except:
      print("cal_PrimaryCount")
      print(tmp_labeled)
    continue
 except:
   print("cal_PrimaryCount")
   print(Labeled[index_labeled])
   print(Tag[index_Tag + 1])
   print("..........")

 return Num
def extract_title(Tag):
    Label = []
    for index in range(len(Tag)):
        tmp_tag = Tag[index]
        ii = tmp_tag.find(":")
        Label.append(tmp_tag[0:ii])
    return Label
def dedupe(Traits):
# filename2 = "G:/Root/Total/Root_trait(Total)/Root measurements._1.csv"
# Traits = pd.read_csv(filename2)
 try:
     items = Traits["Labeled"]
     Labeled = items
     tmp_num = 1
     tmp_num1 = 0
     i = 0
     Item = []
     for index in range(len(Labeled)):
       tmp_labeled = Labeled[index]
       index_1 = index + 1
       tmp_labeled_1 = Labeled.iloc[index_1]
       if (tmp_labeled == tmp_labeled_1):
         continue
       else:
        tmp_item = tmp_labeled
        Item.append(tmp_item)
 except:
     print("dedupe")
     print(tmp_labeled)
     print("..........")
 return Item


def read_xml(filepath,savepath):
    print("start")
    # filepath = "G:/Root_henan_shi/data/result"
    # 修改文件名后缀
    dtype_xml(filepath)
    filename = os.listdir(filepath)
    file_root = []
    root_trait = []
    for i in filename:
        portion = os.path.splitext(i)
        try:
            if  portion[1] == ".xml":
                file_root.append(filepath + '/' + i)
            else:
                print(i)
        except:
            print(i)
    for path in file_root:
        # print(path)
        tree = ET.parse(path)
        root = tree.getroot()
        root_Tag = root[0][6].text
        Root_point = []
        coordinate_y = []
        coordinate_x = []
        coordinate_yy = []
        coordinate_xx = []
        max_coordinate_x = []
        min_corrdinate_x = []
        max_coordinate_y = []
        min_corrdinate_y = []
        tmp_root_trait = []
        tol_max_depth = []
        tol_max_width = []
        tol_Root = []
        tmp_judge = []
        for child in root.iter('root'):
            try:
                tmp_root_label = child.get('ID')
                for child_root in child.iter('point'):
                    tmp_coordinate_x = float(child_root.get('x'))
                    tmp_coordinate_y = float(child_root.get('y'))
                    coordinate_xx.append(tmp_coordinate_x)
                    coordinate_yy.append(tmp_coordinate_y)
                tmp_root_trait.append(root_Tag)
                tmp_root_trait.append(tmp_root_label)
                tmp_root_trait.append(coordinate_xx)
                tmp_root_trait.append(coordinate_yy)
                root_trait.append(tmp_root_trait)
                tmp_root_trait = []
                coordinate_xx = []
                coordinate_yy = []
            except:
                print("..........")
                print(tmp_root_label)
                print("Start")
    # Tol_root_trait = pd.DataFrame(data = root_trait)
    # Tol_root_trait.to_csv("H:/root_Zhixin/root_trait_xx_yy.csv", encoding = 'gbk')

    # # TT = []
    # # for tt in range(len(root_trait)):
    # #     T = root_trait[tt][0]
    # #     TT.append(T)
    # # a = dedupe(TT)
    #
    #
    #
    #
    xx = []
    yy = []
    tol_trait = []
    Tol_trait = []
    Tol_root = []
    tmp_tol_trait = []
    tmp_Tol_root = []
    tmp_xx = []
    tmp_yy = []
    tmp_XX = []
    tmp_YY = []
    max_width = []
    max_depth = []
    tmp_trait = []
    tmp_p = 1
    i_p = 0
    XX = []
    YY = []
    for index in range(len(root_trait)):
        tmp_index = root_trait[index][0]
        index_1 = index + 1
        try:
            tmp_index_1 = root_trait[index_1][0]
            if (tmp_index == tmp_index_1):
                tmp_p += 1
            else:
                 for jj in range(i_p, tmp_p):
                    XX.append(root_trait[jj][2])
                    YY.append(root_trait[jj][3])
                 for tmp_NUM in range(len(XX)):
                     for tmp_NUM_1 in range(len(XX[tmp_NUM])):
                         tmp_XX.append(XX[tmp_NUM][tmp_NUM_1])
                         tmp_YY.append(YY[tmp_NUM][tmp_NUM_1])
                 Max_XX = max(tmp_XX)
                 Min_XX = min(tmp_XX)
                 Max_YY = max(tmp_YY)
                 Min_YY = min(tmp_YY)
                 Max_width = Max_YY - Min_YY
                 Max_depth = Max_XX - Min_XX
                 tmp_Tol_root.append(tmp_index)
                 tmp_Tol_root.append(Max_width)
                 tmp_Tol_root.append(Max_depth)
                 Tol_root.append(tmp_Tol_root)
                 if i_p == 12946:
                     print("Focus")
                 if i_p == 39469:
                     print("Focus")
                 if i_p == 50976:
                     print("Focus")
                 for tmp_ii in range(i_p,tmp_p):
                     tmp = root_trait[tmp_ii]
                     tmp_trait.append(tmp)
                 for ii in range(len(tmp_trait)):
                        tmp_item = tmp_trait[ii][1]
                        ii_1 = ii + 1
                        if (len(tmp_trait)) == 1:
                            tmp_judge = tmp_item[0]
                            xx.append(tmp_trait[ii][2])
                            yy.append(tmp_trait[ii][3])
                            for tmp_num in range(len(xx[0])):
                                tmp_xx.append(xx[0][tmp_num])
                                tmp_yy.append(yy[0][tmp_num])
                            max_xx = max(tmp_xx)
                            min_xx = min(tmp_xx)
                            max_yy = max(tmp_yy)
                            min_yy = min(tmp_yy)
                            tmp_max_width = max_yy - min_yy
                            tmp_max_depth = max_xx - min_xx
                            # max_depth.append(tmp_max_depth)
                            # max_width.append(tmp_max_width)
                            tmp_tol_trait.append(tmp_index)
                            tmp_tol_trait.append(tmp_judge)
                            tmp_tol_trait.append(tmp_max_width)
                            tmp_tol_trait.append(tmp_max_depth)
                            tol_trait.append(tmp_tol_trait)
                            max_width = []
                            tmp_xx = []
                            tmp_yy = []
                            max_depth = []
                            tmp_tol_trait = []
                            xx = []
                            yy = []

                        xx.append(tmp_trait[ii][2])
                        yy.append(tmp_trait[ii][3])
                        try:
                            tmp_item_1 = tmp_trait[ii_1][1]
                            tmp_judge = tmp_item[0:3]
                            tmp_judge_1 = tmp_item_1[0:3]
                            if tmp_item[1] != '.':
                                    tmp_judge = tmp_item[0:5]
                                    tmp_judge_1 = tmp_item_1[0:5]
                            if (tmp_judge == tmp_judge_1):
                                    xx.append(root_trait[ii_1][2])
                                    yy.append(root_trait[ii_1][3])
                            else:
                                    for tmp_num in range(len(xx[0])):
                                        tmp_xx.append(xx[0][tmp_num])
                                        tmp_yy.append(yy[0][tmp_num])
                                    max_xx = max(tmp_xx)
                                    min_xx = min(tmp_xx)
                                    max_yy = max(tmp_yy)
                                    min_yy = min(tmp_yy)
                                    tmp_max_width = max_yy - min_yy
                                    tmp_max_depth = max_xx - min_xx
                                    # max_depth.append(tmp_max_depth)
                                    # max_width.append(tmp_max_width)
                                    tmp_tol_trait.append(tmp_index)
                                    tmp_tol_trait.append(tmp_judge)
                                    tmp_tol_trait.append(tmp_max_width)
                                    tmp_tol_trait.append(tmp_max_depth)
                                    tol_trait.append(tmp_tol_trait)
                                    max_width = []
                                    tmp_xx = []
                                    tmp_yy = []
                                    max_depth = []
                                    tmp_tol_trait = []
                                    xx = []
                                    yy = []
                        except:
                            # I_ii = ii - 1
                            # tmp_I_item = tmp_trait[I_ii][1]
                                tmp_judge = tmp_item[0:3]
                            # I_tmp_judge = tmp_I_item[0:3]
                            # if tmp_item[1] != '.':
                            #     tmp_judge = tmp_item[0:5]
                            #     I_tmp_judge = tmp_I_item[0:5]
                            # if tmp_judge == I_tmp_judge:
                            #     continue
                            # else:
                                for tmp_num in range(len(xx[0])):
                                    tmp_xx.append(xx[0][tmp_num])
                                    tmp_yy.append(yy[0][tmp_num])
                                max_xx = max(tmp_xx)
                                min_xx = min(tmp_xx)
                                max_yy = max(tmp_yy)
                                min_yy = min(tmp_yy)
                                tmp_max_width = max_yy - min_yy
                                tmp_max_depth = max_xx - min_xx
                                # max_depth.append(tmp_max_depth)
                                # max_width.append(tmp_max_width)
                                tmp_tol_trait.append(tmp_index)
                                tmp_tol_trait.append(tmp_judge)
                                tmp_tol_trait.append(tmp_max_width)
                                tmp_tol_trait.append(tmp_max_depth)
                                tol_trait.append(tmp_tol_trait)
                                max_width = []
                                tmp_xx = []
                                tmp_yy = []
                                max_depth = []
                                tmp_tol_trait = []
                                xx = []
                                yy = []

                 i_p = tmp_p
                 tmp_p += 1
                 tmp_XX = []
                 tmp_YY = []
                 tmp_Tol_root = []
                 XX = []
                 YY = []
                 tmp_trait = []

        except:
                tmp_p = index_1
                for jj in range(i_p, tmp_p):
                    XX.append(root_trait[jj][2])
                    YY.append(root_trait[jj][3])
                for tmp_NUM in range(len(XX[0])):
                    tmp_XX.append(XX[0][tmp_NUM])
                    tmp_YY.append(YY[0][tmp_NUM])
                Max_XX = max(tmp_XX)
                Min_XX = min(tmp_XX)
                Max_YY = max(tmp_YY)
                Min_YY = min(tmp_YY)
                Max_width = Max_YY - Min_YY
                Max_depth = Max_XX - Min_YY
                tmp_Tol_root.append(tmp_index)
                tmp_Tol_root.append(Max_width)
                tmp_Tol_root.append(Max_depth)
                Tol_root.append(tmp_Tol_root)
                for ii in range(i_p, tmp_p):
                    tmp_item = root_trait[ii][1]
                    ii_1 = ii + 1
                    try:
                        tmp_item_1 = root_trait[ii_1][1]
                        xx.append(root_trait[ii][2])
                        yy.append(root_trait[ii][3])
                        tmp_judge = tmp_item[0]
                        tmp_judge_1 = tmp_item_1[0]
                        if tmp_item[1] != '.':
                            tmp_judge = tmp_item[0:2]
                            tmp_judge_1 = tmp_item_1[0:2]
                        if (tmp_judge == tmp_judge_1):
                                xx.append(root_trait[ii_1][2])
                                yy.append(root_trait[ii_1][3])
                        else:
                                for tmp_num in range(len(xx[0])):
                                    tmp_xx.append(xx[0][tmp_num])
                                    tmp_yy.append(yy[0][tmp_num])
                                max_xx = max(tmp_xx)
                                min_xx = min(tmp_xx)
                                max_yy = max(tmp_yy)
                                min_yy = min(tmp_yy)
                                tmp_max_width = max_yy - min_yy
                                tmp_max_depth = max_xx - min_xx
                                # max_depth.append(tmp_max_depth)
                                # max_width.append(tmp_max_width)
                                tmp_tol_trait.append(tmp_index)
                                tmp_tol_trait.append(tmp_judge)
                                tmp_tol_trait.append(tmp_max_width)
                                tmp_tol_trait.append(tmp_max_depth)
                                tol_trait.append(tmp_tol_trait)
                                max_width = []
                                max_depth = []
                                tmp_tol_trait = []
                                xx = []
                                yy = []
                    except:
                            xx.append(root_trait[ii][2])
                            yy.append(root_trait[ii][3])
                            for tmp_num in range(len(xx[0])):
                                tmp_xx.append(xx[0][tmp_num])
                                tmp_yy.append(yy[0][tmp_num])
                            max_xx = max(tmp_xx)
                            min_xx = min(tmp_xx)
                            max_yy = max(tmp_yy)
                            min_yy = min(tmp_yy)
                            tmp_max_width = max_yy - min_yy
                            tmp_max_depth = max_xx - min_xx
                            # max_depth.append(tmp_max_depth)
                            # max_width.append(tmp_max_width)
                            tmp_tol_trait.append(tmp_index)
                            tmp_tol_trait.append(tmp_item[0])
                            tmp_tol_trait.append(tmp_max_width)
                            tmp_tol_trait.append(tmp_max_depth)
                            tol_trait.append(tmp_tol_trait)
                i_p = tmp_p
                tmp_p += 1
                tmp_XX = []
                tmp_YY = []
                tmp_Tol_root = []
                XX = []
                YY = []
                print('.....')
                print(tmp_index)
    print(" tol_trait over")

    # Tol_root_trait = pd.DataFrame(data = Tol_root)
    # Tol_root_trait.to_csv("H:/root_Zhixin/Tol_root_trait.csv", encoding = 'gbk')

    root_trait = pd.DataFrame(data=tol_trait)
    tmp_wdRatio = (root_trait.iloc[:,2])/(root_trait.iloc[:,3])
    root_trait.insert(4, 'wdRatio', tmp_wdRatio)
    root_trait.rename(columns={0: "Name", 1: "Index", 2: "width", 3: "depth",4: "wdRatio"})
    # root_trait.to_csv(savepath, encoding = 'gbk')
    return root_trait

# 计算主根各自深度、宽度
def cal_WD_ratio(tol_trait,savepath):
    # filename = "root_trait(primary_root_depart).csv"
    # tol_trait = pd.read_csv(filename)
    tmp_p_tt = 1
    i_p_tt = 0
    tmp_Width = []
    tmp_Depth = []
    Average_Width = []
    Max_Width = []
    Min_Width = []
    Average_Depth = []
    Max_Depth = []
    Min_Depth = []
    tmp_WD_ratio = []
    tmp_WD_ratio_1 = []
    Tol_WD = []
    tmp_Tol_WD = []
    Label = list(tol_trait[0])
    # Width = list(tol_trait['width'])
    # Depth = list(tol_trait['depth'])
    WD_Ratio = list(tol_trait["wdRatio"])
    for tt in range(len(tol_trait)):
        tmp_tt = Label[tt]
        tt_1 = tt + 1
        try:
            tmp_tt_1 = Label[tt_1]
            if (tmp_tt == tmp_tt_1):
                tmp_p_tt += 1
            else:
                for n in range(i_p_tt, tmp_p_tt):
                    tmp_WD_ratio_1.append(WD_Ratio[n])
                for ii in range(len(tmp_WD_ratio_1)):
                    tmp_tmp = float(tmp_WD_ratio_1[ii])
                    tmp_WD_ratio.append(tmp_tmp)
                    # tmp_Width.append(Width[n])
                    # tmp_Depth.append(Depth[n])
                # Average_Width = np.mean(tmp_Width)
                # Max_width = max(tmp_Width)
                # Min_Width = min(tmp_Width)
                # Average_Depth = np.mean(tmp_Depth)
                # Max_Depth = max(tmp_Depth)
                # Min_Depth = min(tmp_Depth)
                # tmp_Tol_WD.append(tmp_tt)
                # tmp_Tol_WD.append(Average_Width)
                # tmp_Tol_WD.append(Max_width)
                # tmp_Tol_WD.append(Min_Width)
                # tmp_Tol_WD.append(Average_Depth)
                # tmp_Tol_WD.append(Max_Depth)
                # tmp_Tol_WD.append(Min_Depth)
                Average_WD_ratio = np.mean(tmp_WD_ratio)
                Max_WD_ratio = max(tmp_WD_ratio)
                Min_WD_ratio = min(tmp_WD_ratio)
                tmp_Tol_WD.append(tmp_tt)
                tmp_Tol_WD.append(Average_WD_ratio)
                tmp_Tol_WD.append(Max_WD_ratio)
                tmp_Tol_WD.append(Min_WD_ratio)
                Tol_WD.append(tmp_Tol_WD)
                tmp_Width = []
                tmp_Depth = []
                tmp_WD_ratio = []
                tmp_WD_ratio_1 = []
                tmp_Tol_WD = []
                i_p_tt = tmp_p_tt
                tmp_p_tt += 1
        except:
            print(".......")
            print(tmp_tt)
            tmp_p_tt = tt_1
            for nn in range(i_p_tt, tmp_p_tt):
                tmp_WD_ratio_1.append(WD_Ratio[nn])
            for iii in range(len(tmp_WD_ratio_1)):
                tmp_tmp = float(tmp_WD_ratio_1[iii])
                tmp_WD_ratio.append(tmp_tmp)
            # Average_Width = np.mean(tmp_Width)
            # Max_width = max(tmp_Width)
            # Min_Width = min(tmp_Depth)
            # Average_Depth = np.mean(tmp_Depth)
            # Max_width = max(tmp_Depth)
            # Min_Depth = min(tmp_Depth)
            # tmp_Tol_WD.append(tmp_tt)
            # tmp_Tol_WD.append(Average_Width)
            # tmp_Tol_WD.append(Max_width)
            # tmp_Tol_WD.append(Min_Width)
            # tmp_Tol_WD.append(Average_Depth)
            # tmp_Tol_WD.append(Max_Depth)
            # tmp_Tol_WD.append(Min_Depth)
            # Tol_WD.append(tmp_Tol_WD)
            Average_WD_ratio = np.mean(tmp_WD_ratio)
            Max_WD_ratio = max(tmp_WD_ratio)
            Min_WD_ratio = min(tmp_WD_ratio)
            tmp_Tol_WD.append(tmp_tt)
            tmp_Tol_WD.append(Average_WD_ratio)
            tmp_Tol_WD.append(Max_WD_ratio)
            tmp_Tol_WD.append(Min_WD_ratio)
            Tol_WD.append(tmp_Tol_WD)
            tmp_Width = []
            tmp_Depth = []
            tmp_WD_ratio = []
            tmp_WD_ratio_1 = []
            tmp_Tol_WD = []
            print('.....')
    Tol_WD_trait = pd.DataFrame(data=Tol_WD)
    # Tol_WD_trait.to_csv("H:/root_Zhixin/Tol_WD_trait.csv", encoding = 'gbk')
    Tol_WD_trait.to_csv(savepath, encoding='gbk')
    return Tol_WD_trait


def cal_WD(data,savepath):
    Index_name = ["Name", "Index", "width", "depth"]
    Name = list(data[0])
    Width = list(data[2])
    Depth = list(data[3])
    tmp_p_tt = 1
    i_p_tt = 0
    tmp_Width = []
    tmp_Depth = []
    Tol_WD = []
    tmp_Tol_WD = []
    for tt in range(len(Name)):
        tmp_tt = Name[tt]
        tt_1 = tt + 1
        try:
            tmp_tt_1 = Name[tt_1]
            if (tmp_tt == tmp_tt_1):
                tmp_p_tt += 1
            else:
                for n in range(i_p_tt, tmp_p_tt):
                    tmp_Width.append(Width[n])
                    tmp_Depth.append(Depth[n])
                Average_Width = np.mean(tmp_Width)
                Max_width = max(tmp_Width)
                Min_Width = min(tmp_Width)
                Average_Depth = np.mean(tmp_Depth)
                Max_Depth = max(tmp_Depth)
                Min_Depth = min(tmp_Depth)
                tmp_Tol_WD.append(tmp_tt)
                tmp_Tol_WD.append(Average_Width)
                tmp_Tol_WD.append(Max_width)
                tmp_Tol_WD.append(Min_Width)
                tmp_Tol_WD.append(Average_Depth)
                tmp_Tol_WD.append(Max_Depth)
                tmp_Tol_WD.append(Min_Depth)
                Tol_WD.append(tmp_Tol_WD)
                tmp_Width = []
                tmp_Depth = []
                tmp_Tol_WD = []
                i_p_tt = tmp_p_tt
                tmp_p_tt += 1
        except:
            print(".......")
            print(tmp_tt)
            tmp_p_tt = tt_1
            for n in range(i_p_tt, tmp_p_tt):
                tmp_Width.append(Width[n])
                tmp_Depth.append(Depth[n])
            Average_Width = np.mean(tmp_Width)
            Max_width = max(tmp_Width)
            Min_Width = min(tmp_Width)
            Average_Depth = np.mean(tmp_Depth)
            Max_Depth = max(tmp_Depth)
            Min_Depth = min(tmp_Depth)
            tmp_Tol_WD.append(tmp_tt)
            tmp_Tol_WD.append(Average_Width)
            tmp_Tol_WD.append(Max_width)
            tmp_Tol_WD.append(Min_Width)
            tmp_Tol_WD.append(Average_Depth)
            tmp_Tol_WD.append(Max_Depth)
            tmp_Tol_WD.append(Min_Depth)
            Tol_WD.append(tmp_Tol_WD)
            print('.....')
            print(tmp_tt)
    Tol_WD = pd.DataFrame(data=Tol_WD)
    # Tol_WD_trait.to_csv("H:/root_Zhixin/Tol_WD_trait.csv", encoding = 'gbk')
    Tol_WD.to_csv(savepath, encoding='gbk')
    return Tol_WD

def Tol_WD(filepath,savepath):
    print("start")
    # filepath = "G:/Root_henan_shi/data/result"
    # 修改文件名后缀
    dtype_xml(filepath)
    filename = os.listdir(filepath)
    file_root = []
    root_trait = []
    for i in filename:
        portion = os.path.splitext(i)
        try:
            if portion[1] == ".xml":
                file_root.append(filepath + '/' + i)
            else:
                print(i)
        except:
            print(i)

    tol_Root = []
    for path in file_root:
        try:
            print(path)
            tree = ET.parse(path)
            root = tree.getroot()
            roo_Tag = root[0][6].text
            Root_point = []
            coordinate_y = []
            coordinate_x = []
            coordinate_yy = []
            coordinate_xx = []
            max_coordinate_x = []
            min_corrdinate_x = []
            max_coordinate_y = []
            min_corrdinate_y = []
            tmp_root_trait = []
            tol_max_depth = []
            tol_max_width = []
            root_trait = []
            tmp_tol_Root = []
            # print(root)
            # print(root.tag)
            # print(root.attrib)
            for child in root.iter('root'):
                tmp_root_label = child.get('ID')
                # tmp_root_label = int(tmp_root_label)
                for child_root in child.iter('point'):
                    tmp_coordinate_xx = float(child_root.get('x'))
                    tmp_coordinate_yy = float(child_root.get('y'))
                    coordinate_xx.append(tmp_coordinate_xx)
                    coordinate_yy.append(tmp_coordinate_yy)
            tmp_max_tol_x = max(coordinate_xx)
            tmp_min_tol_x = min(coordinate_xx)
            tmp_depth = tmp_max_tol_x - tmp_min_tol_x
            tmp_max_tol_y = max(coordinate_yy)
            tmp_min_tol_y = min(coordinate_yy)
            tmp_width = tmp_max_tol_y - tmp_min_tol_y
            tmp_tol_Root.append(roo_Tag)
            tmp_tol_Root.append(tmp_width)
            tmp_tol_Root.append(tmp_depth)
            tol_Root.append(tmp_tol_Root)
        except:
            print("Wrong")
            print(path)
            print("...............")
    Tol_WD_trait = pd.DataFrame(tol_Root)
    Tol_WD_trait.to_csv(savepath,encoding='gbk')
    print("over")
    return Tol_WD_trait



def data_process(Labeled_plant,Labeled_root,filename_plant,filename_root,filepath_xml,savepath,savepath_xxyypoint,savepath_Primal_WD_depart,savepath_Tol_WD,savepath_filename_cal_WD_ratio):
    # filename1 = "G:/Root/Total/Root_trait(Total)/Plant measurements._1.csv"
    #
    # filename2 = "G:/Root/Total/Root_trait(Total)/Root measurements._1.csv"
    text = []
    text_1 = []
    text = ['text', 'text', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    # pd.DataFrame(data=text)
    Traits = pd.read_csv(filename_plant)
    Traits.insert(1,'Labeled',Labeled_plant[0])
    Traits.loc[np.shape(Traits)[0] + 1] = ['text', 'text', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    TTraits = pd.DataFrame(Traits)
    TTraits.to_csv("F:/Root_henan_shi/data5/result/plant measurements_1.csv", encoding='gbk')
    Traits_1 = pd.read_csv(filename_root)
    Traits_1.insert(1,'Labeled',Labeled_root[0])
    Traits_1.loc[np.shape(Traits_1)[0] + 1] = ['text', 'text', 0, 0, 0, 0, 0, 0, 0]
    TTraits_1 = pd.DataFrame(Traits_1)
    TTraits_1.to_csv("F:/Root_henan_shi/data5/result/root measurements_1.csv", encoding='gbk')
    Traits = pd.read_csv("F:/Root_henan_shi/data5/result/plant measurements_1.csv")
    Traits_1 = pd.read_csv("F:/Root_henan_shi/data5/result/root measurements_1.csv")
    Traits.info()
    Traits_1.info()
    # Labeled = extract_title(Traits['Tag'])
    # Labeled = pd.DataFrame(index=Traits['Tag'], data=Labeled)
    # Labeled.to_csv("G:/Root_henan_shi/data2/result/label.csv", encoding='gbk')
    # Labeled_1 = extract_title(Traits_1['Tag'])
    # Labeled_1 = pd.DataFrame(index=Traits_1['Tag'], data=Labeled_1)
    # Labeled_1.to_csv("G:/Root_henan_shi/data2/result/Labeled_1.csv", encoding='gbk')
    Title = dedupe(Traits)
    # 此时返回的数据类型是元组
    # Angle
    _, Avearge_PrimaryTipAngle, Max_PrimaryTipAngle, Min_PrimaryTipAngle = cal_PrimaryTrait(Traits_1, 'Tag', 'Labeled',
                                                                                            'Tip Angle')
    _, Average_PrimaryEmergenceAngle, Max_PrimaryEmergenceAngle, Min_PrimaryEmergenceAngle = cal_PrimaryTrait(Traits_1,
                                                                                                              'Tag',
                                                                                                              'Labeled',
                                                                                                              'Emergence Angle')
    _, Average_LateralTipAngle, Max_LateralTipAngle, Min_LateralTipAngle = cal_LataralTrait(Traits_1, 'Tag', 'Labeled',
                                                                                            'Tip Angle')
    _, Average_LateralEmergenceAngle, Max_LateralEmergenceAngle, Min_LateralEmergenceAngle = cal_LataralTrait(Traits_1,
                                                                                                              'Tag',
                                                                                                              'Labeled',
                                                                                                              'Emergence Angle')
    # Length
    Total_PrimaryLength, Average_PrimaryLength, Max_PrimaryLength, Min_PrimaryLength = cal_PrimaryTrait(Traits_1, 'Tag',
                                                                                                        'Labeled',
                                                                                                        'Total Length')
    Total_LateralLength, Average_LateralLength, Max_LateralLength, Min_LateralLength = cal_LataralTrait(Traits_1, 'Tag',
                                                                                                        'Labeled',
                                                                                                        'Total Length')
    # Count
    PrimaryCount = cal_PrimaryCount(Traits_1, 'Tag', 'Labeled')
    AverageLPCount, MaxLPCount, MinLPCount = cal_LPcount(Traits_1, 'Tag', 'Labeled')
    # Tortuosity
    _, Average_PrimaryTortuosity, Max_PrimaryTortuosity, Min_PrimaryTortuosity = cal_PrimaryTrait(Traits_1, 'Tag',
                                                                                                  'Labeled',
                                                                                                  'Tortuosity')
    _, Average_LateralTortuosity, Max_LateralTortuosity, Min_LateralTortuosity = cal_LataralTrait(Traits_1, 'Tag',
                                                                                                  'Labeled',
                                                                                                  'Tortuosity')
    # 注意格式 这里是series
    trait_labeled = Traits['Labeled']
    trait_PrimaryTipAngle = Traits['Average Primary Tip Angle']
    trait_PrimaryEmergenceAngle = Traits['Average Primary Emergence Angle']
    trait_LateralTipAngle = Traits['Average Lateral Tip Angle']
    trait_LateralEmergenceAngle = Traits['Average Lateral Emergence Angle']
    trait_Length = Traits['Total Length']
    trait_PrimaryLength = Traits['Average Length - Primary roots']
    trait_LateralLength = Traits['Average Length - Lateral roots']
    trait_LateralCount = Traits['Lateral Root Count']
    trait_Tortuosity = Traits['Average Tortuosity']
    trait_MaxWidth = Traits['Maximum Width']
    trait_MaxDepth = Traits['Maximum Depth']
    trait_DWRatio = Traits['Width / Depth Ratio']
    trait_ConvexHull = Traits['Convex Hull']
    tmp_num = 1
    i = 0
    Total_RootLength = []
    Average_PrimaryWidth = []
    Max_PrimaryWidth = []
    Min_PrimaryWidth = []
    Average_PrimaryDepth = []
    Max_PrimaryDepth = []
    Min_PrimaryDepth = []
    Average_PrimaryWDRatio = []
    Max_PrimaryWDRatio = []
    Min_PrimaryWDRatio = []
    Average_PrimaryConexHull = []
    Max_PrimaryConexHull = []
    Min_PrimaryConexHull = []
    LateralRootCount = []
    try:
        for index in range(len(trait_labeled)):
            tmp_labeled = trait_labeled[index]
            index_1 = index + 1
            try:
                tmp_labeled_1 = trait_labeled.iloc[index_1]
                if (tmp_labeled == tmp_labeled_1):
                    tmp_num += 1
                else:
                    tmp_Total_RootLength = sum(trait_Length[i:tmp_num])
                    tmp_LateralRootCount = sum(trait_LateralCount[i:tmp_num])
                    tmp_Average_PrimaryWidth = np.mean(trait_MaxWidth[i:tmp_num])
                    tmp_Max_PrimaryWidth = max(trait_MaxWidth[i:tmp_num])
                    tmp_Min_PrimaryWidth = min(trait_MaxWidth[i:tmp_num])
                    tmp_Average_PrimaryDepth = np.mean(trait_MaxDepth[i:tmp_num])
                    tmp_Max_PrimaryDepth = max(trait_MaxDepth[i:tmp_num])
                    tmp_Min_PrimaryDepth = min(trait_MaxDepth[i:tmp_num])
                    tmp_Average_PrimaryWDRatio = np.mean(trait_DWRatio[i:tmp_num])
                    tmp_Max_PrimaryWDRatio = max(trait_DWRatio[i:tmp_num])
                    tmp_Min_PrimaryWDRatio = min(trait_DWRatio[i:tmp_num])
                    PrimaryConexHull = sum(trait_ConvexHull[i:tmp_num])
                    tmp_PrimaryConexHull = cal_LataralCount(trait_ConvexHull, i, tmp_num)
                    if tmp_PrimaryConexHull == 0:
                        tmp_Average_PrimaryConexHull = 0
                    else:
                        tmp_Average_PrimaryConexHull = PrimaryConexHull / tmp_PrimaryConexHull
                    tmp_Max_PrimaryConexHull = max(trait_ConvexHull[i:tmp_num])
                    tmp_Min_PrimaryConexHull = min(trait_ConvexHull[i:tmp_num])
                    i = tmp_num
                    tmp_num += 1
                    #
                    Total_RootLength.append(tmp_Total_RootLength)
                    LateralRootCount.append(tmp_LateralRootCount)
                    # AverageLPCount.append(tmp_AverageLPCount)
                    # MaxLPCount.append(tmp_MaxLPCount)
                    # MinLPCount.append(tmp_MinLPCount)
                    Average_PrimaryWidth.append(tmp_Average_PrimaryWidth)
                    Max_PrimaryWidth.append(tmp_Max_PrimaryWidth)
                    Min_PrimaryWidth.append(tmp_Min_PrimaryWidth)
                    Average_PrimaryDepth.append(tmp_Average_PrimaryDepth)
                    Max_PrimaryDepth.append(tmp_Max_PrimaryDepth)
                    Min_PrimaryDepth.append(tmp_Min_PrimaryDepth)
                    Average_PrimaryWDRatio.append(tmp_Average_PrimaryWDRatio)
                    Max_PrimaryWDRatio.append(tmp_Max_PrimaryWDRatio)
                    Min_PrimaryWDRatio.append(tmp_Min_PrimaryWDRatio)
                    Average_PrimaryConexHull.append(tmp_Average_PrimaryConexHull)
                    Max_PrimaryConexHull.append(tmp_Max_PrimaryConexHull)
                    Min_PrimaryConexHull.append(tmp_Min_PrimaryConexHull)
            except:
                print("extra_trait_wrong")
    except:
        print("cal_extra_trait")
        print(tmp_labeled)
    RootTrait = []
    RootTrait.append(Title)
    RootTrait.append(Avearge_PrimaryTipAngle)
    RootTrait.append(Max_PrimaryTipAngle)
    RootTrait.append(Min_PrimaryTipAngle)
    RootTrait.append(Average_PrimaryEmergenceAngle)
    RootTrait.append(Max_PrimaryEmergenceAngle)
    RootTrait.append(Min_PrimaryEmergenceAngle)
    RootTrait.append(Average_LateralTipAngle)
    RootTrait.append(Max_LateralTipAngle)
    RootTrait.append(Min_LateralTipAngle)
    RootTrait.append(Average_LateralEmergenceAngle)
    RootTrait.append(Max_LateralEmergenceAngle)
    RootTrait.append(Min_LateralEmergenceAngle)
    RootTrait.append(Total_RootLength)
    RootTrait.append(Total_PrimaryLength)
    RootTrait.append(Average_PrimaryLength)
    RootTrait.append(Max_PrimaryLength)
    RootTrait.append(Min_PrimaryLength)
    RootTrait.append(Total_LateralLength)
    RootTrait.append(Average_LateralLength)
    RootTrait.append(Max_LateralLength)
    RootTrait.append(Min_LateralLength)
    RootTrait.append(PrimaryCount)
    RootTrait.append(LateralRootCount)
    RootTrait.append(AverageLPCount)
    RootTrait.append(MaxLPCount)
    RootTrait.append(MinLPCount)
    # RootTrait.append(Average_PrimaryWidth)
    # RootTrait.append(Max_PrimaryWidth)
    # RootTrait.append(Min_PrimaryWidth)
    # RootTrait.append(Average_PrimaryDepth)
    # RootTrait.append(Max_PrimaryDepth)
    # RootTrait.append(Min_PrimaryDepth)
    RootTrait.append(Average_PrimaryTortuosity)
    RootTrait.append(Max_PrimaryTortuosity)
    RootTrait.append(Min_PrimaryTortuosity)
    RootTrait.append(Average_LateralTortuosity)
    RootTrait.append(Max_LateralTortuosity)
    RootTrait.append(Min_LateralTortuosity)
    # RootTrait.append(Average_PrimaryWDRatio)
    # RootTrait.append(Max_PrimaryWDRatio)
    # RootTrait.append(Min_PrimaryWDRatio)
    RootTrait.append(Average_PrimaryConexHull)
    RootTrait.append(Max_PrimaryConexHull)
    RootTrait.append(Min_PrimaryConexHull)
    root_trait = read_xml(filepath_xml, savepath_xxyypoint)
    Primal_WD_depart = cal_WD(root_trait, savepath_Primal_WD_depart)
    Tol_WDth = Tol_WD(filepath_xml, savepath_Tol_WD)
    cal_WDth_ratio = cal_WD_ratio(root_trait, savepath_filename_cal_WD_ratio)
    tmp_wd_trait = np.concatenate([Primal_WD_depart.iloc[:,1:7],Tol_WDth.iloc[:,1:3],cal_WDth_ratio.iloc[:,1:4]],axis=1)
    wd_trait = pd.DataFrame(data=tmp_wd_trait)
    wd_trait.to_csv("F:/Root_henan_shi/data5/result/Root tmp_wd_trait.csv", encoding='gbk')
    tmp_root_trait = pd.read_csv("F:/Root_henan_shi/data5/result/Root tmp_wd_trait.csv")
    list = []
    # root_trait=(tmp_root_trait.stack()).unstack(0)
    Tmp_root_trait = pd.DataFrame(tmp_root_trait.values.T, index=tmp_root_trait.columns, columns=tmp_root_trait.index)
    Root_trait = np.array(Tmp_root_trait).tolist()
    Tol_RootTrait = RootTrait + (Root_trait[1:np.shape(root_trait)[0]])
    Index_name = ["Name", "Average Primary Tip Angle", "Maximum Primary Tip Angle", "Minimum Primary Tip Angle",
                  "Average Primary Emergence Angle", "Maximum Primary Emergence Angle",
                  "Minimum Primary Emergence Angle", \
                  "Average Lateral Tip Angle", "Maximum Lateral Tip Angle", "Minimum Lateral Tip Angle",
                  "Average Lateral Emergence Angle", "Maximum Lateral Emergence Angle",
                  "Minimum Lateral Emergence Angle", "Total Roots Length", \
                  "Total Primary Root Length", "Average Primary Root Length", "Maximum Primary Root Length",
                  "Minimum Primary Root Length", "Total Lateral Root Length", "Average Lateral Root Length",
                  "Maximum Lateral Root Length", \
                  "Minimum Lateral Root Length", "Primary Root Count", "Lateral Root Count",
                  "Average Lateral Root Count-Primary Root Count", "Maximum Lateral Root Count-Primary Root Count",
                  "Minimum Lateral Root Count-Primary Root Count", \
                  # "Average Primary Root Maximum Width", "Maximum Primary Root Maximum Width",
                  # "Minimum Primary Root Maximum Width ", "Average Primary Root Maximum Depth",
                  # "Maximum Primary Root Maximum Depth", "Minimum Primary Root Maximum Depth", \
                  "Average Primary Root Tortuosity", "Maximum Primary Root Tortuosity",
                  "Minimum Primary Root Tortuosity", "Average Lateral Root Tortuosity",
                  "Maximum Lateral Root Tortuosity", "Minimum Lateral Root Tortuosity",
                  # "Average Primary Root of  Width / Depth Ratio", \
                  # "Maximum Primary Root of Width / Depth Ratio", "Minimum Primary Root of  Width / Depth Ratio",
                  "Average Primary Root Convex Hull", "Maximum Primary Root Convex Hull",
                  "Minimum Primary Root Convex Hull",
                  "Average Primary Root Maximum Width", "Maximum Primary Root Maximum Width","Minimum Primary Root Maximum Width",
                  "Average Primary Root Maximum Depth", "Maximum Primary Root Maximum Depth","Minimum Primary Root Maximum Depth",
                  "Tol_Root Maximum Width", "Tol_Root Maximum Depth",
                  "Average Primary Root of Width / Depth Ratio", "Maximum Primary Root of Width / Depth Ratio", "Minimum Primary Root of Width / Depth Ratio"
                  ]
    Toltrait = pd.DataFrame(index=Index_name, data=Tol_RootTrait)
    Toltrait.to_csv(savepath, encoding='gbk')
    print("over")


#记得在Root measurements.csv、Plant measurements.csv后面加一行Text
filepath_root="F:/Root_henan_shi/data5/result/Root measurements.csv"
savepath_root="F:/Root_henan_shi/data5/result/Root_title.csv"
filepath_plant="F:/Root_henan_shi/data5/result/Plant measurements.csv"
savepath_plant="F:/Root_henan_shi/data5/result/Plant_title.csv"
Labeled_root = listname(filepath_root,savepath_root)
Labeled_plant = listname(filepath_plant,savepath_plant)
filepath_xml = "F:/Root_henan_shi/data5/result"
filename1 = "F:/Root_henan_shi/data5/result/Plant measurements.csv"
filename2 = "F:/Root_henan_shi/data5/result/Root measurements.csv"
# Average Primary Root Maximum Width	Maximum Primary Root Maximum Width 	Minimum Primary Root Maximum Width	Average Primary Root Maximum Depth	Maximum Primary Root Maximum Depth 	Minimum Primary Root Maximum Depth	Tol_Root Maximum Width	Tol_Root Maximum Depth	Average Primary Root of Width / Depth Ratio	Maximum Primary Root of Width / Depth Ratio	Minimum Primary Root of Width / Depth Ratio
savepath = "F:/Root_henan_shi/data5/result/Total_trait2.csv"
savepath_xxyypoint = "F:/Root_henan_shi/data5/result/xxyy_point.csv"
savepath_Primal_WD_depart = "F:/Root_henan_shi/data5/result/Primal_WD_depart.csv"
savepath_Tol_WD = "F:/Root_henan_shi/data5/result/Tol_WD.csv"
savepath_filename_cal_WD_ratio= "F:/Root_henan_shi/data5/result/cal_WD_ratio.csv"
data_process(Labeled_plant,Labeled_root,filename1,filename2,filepath_xml,savepath,savepath_xxyypoint,savepath_Primal_WD_depart,savepath_Tol_WD,savepath_filename_cal_WD_ratio)
# filepath_xml = "G:/Root_henan_shi/data5/result"
# Name Width	Depth	Width / Depth Ratio
# savepath_xxyypoint = "G:/Root_henan_shi/data5/result/xxyy_point.csv"
# tol_trait = read_xml(filepath_xml,savepath_xxyypoint)
# filename3 = "G:/Root_henan_shi/data5/result/root_trait.csv"
# # Average_Width	Max_Width	Min_Width	Average_Depth	Max_Depth	Min_Depth
# savepath_Primal_WD_depart = "G:/Root_henan_shi/data5/result/Primal_WD_depart.csv"
# # root_trait = pd.read_csv(filename3)
# Primal_WD_depart = cal_WD(tol_trait,savepath_Primal_WD_depart)
# filename4 = "G:/Root_henan_shi/data5/result"
# # Width	Depth
# savepath_Tol_WD = "G:/Root_henan_shi/data5/result/Tol_WD.csv"
# Tol_WD = Tol_WD(filename4,savepath_Tol_WD)
# filename5 = "G:/Root_henan_shi/data5/result/root_trait.csv"
# # average_wd	max_wd	min_wd
# savepath_filename_cal_WD_ratio= "G:/Root_henan_shi/data5/result/cal_WD_ratio.csv"
# cal_WD_ratio = cal_WD_ratio(tol_trait,savepath_filename_cal_WD_ratio)




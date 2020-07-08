'''
@Author: your name
@Date: 2020-05-24 08:19:21
@LastEditTime: 2020-05-24 20:50:46
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: \day3\kz.py
'''
# encoding:gbk
# -*- coding: utf-8 -*-
import codecs,csv
import io
import sys
import datetime
sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8')



listData=[]
listCityData = []
listCityName = []
listProvinceName = []

def readFile():
    with open('D:\Study\python\kz\question_input （Python）.csv') as f:
        readers = csv.reader(f)
        header = next(readers)  # 读取第一行每一列的标题
        for i in readers:
            date = datetime.datetime.strptime(i[2],'%m/%d/%Y') 
            i[2] = date
            listData.append(i)

            listCityName.append(i[1])
            listProvinceName.append(i[0])




def analyDataByCity():
    global listCityName
    global listProvinceName
    listCityName = list(set(listCityName))
    listProvinceName =list(set(listProvinceName))
    print(len(listData))
    print(len(listCityName))
    print(len(listProvinceName))
    for data in listData:
        
        
            
            
readFile()
analyDataByCity()
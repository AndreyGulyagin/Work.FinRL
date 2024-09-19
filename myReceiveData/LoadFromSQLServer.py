# инсталируем библиотеки
# numpy
# pandas
# pyodbc
# openpyxl


import numpy as np
# подключаем библиотеку работы с внешними данными
import pandas as pd
from datetime import datetime
import pyodbc
import os

PathData = '..\\DataForex'

# ------------------------------------------------
# функция расчета скольящей средней
def moving_average(series, n):
    k = len(series)
    output = np.zeros(k)
    x = series
    i = n
    while i < k:
        output[i] = np.sum(x[i - 1:i - n - 1:-1]) / n
        i += 1
    return output


# ------------------------------------------------

# ------------------------------------------------
# функция расчета среднего тела свечи
def AvgBody(df_Open, df_Close):
    k = len(df_Open)
    sum = 0
    i = 0
    while i < k:
        sum += abs(df_Open[i] - df_Close[i])
        i += 1
    output = np.full(k, sum / k)
    return output


# ------------------------------------------------

# ------------------------------------------------
# функция поиска патерна Morning Star
def CheckPatternMorningStar(df_Open, df_Close, df_AvgBody, df_MdOC):
    k = len(df_Open)
    output = np.zeros(k)
    i = 3
    while i < k:
        if ((df_Open[i - 2] - df_Close[i - 2] > df_AvgBody[i]) and \
                (abs(df_Close[i - 1] - df_Open[i - 1]) < df_AvgBody[i] / 2) and \
                (df_Close[i - 1] < df_Close[i - 2]) and \
                (df_Open[i - 1] < df_Open[i - 2]) and \
                (df_Close[i] > df_MdOC[i - 2])):
            output[i] = 1
        i += 1
    return output


# ------------------------------------------------


# ------------------------------------------------
# функция поиска патерна Evening Star
def CheckPatternEveningStar(df_Open, df_Close, df_AvgBody, df_MdOC):
    k = len(df_Open)
    output = np.zeros(k)
    i = 3
    while i < k:
        if ((df_Close[i - 2] - df_Open[i - 2] > df_AvgBody[i]) and \
                (abs(df_Close[i - 1] - df_Open[i - 1]) < df_AvgBody[i] / 2) and \
                (df_Close[i - 1] > df_Close[i - 2]) and \
                (df_Open[i - 1] > df_Open[i - 2]) and \
                (df_Close[i] < df_MdOC[i - 2])):
            output[i] = 1
        i += 1
    return output


# ------------------------------------------------
# ------------------------------------------------
# функция поиска патерна Bearish Engulfing
def CheckPatternBearishEngulfing(df_Open, df_Close, df_AvgBody, df_MdOC, df_CloseAvg):
    k = len(df_Open)
    output = np.zeros(k)
    i = 3
    while i < k:
        if ((df_Open[i - 1] < df_Close[i - 1]) and \
                ((df_Open[i] - df_Close[i]) > df_AvgBody[i]) and \
                (df_Close[i] < df_Open[i - 1]) and \
                (df_MdOC[i - 1] > df_CloseAvg[i - 1]) and \
                (df_Open[i] > df_Close[i - 1])):
            output[i] = 1
        i += 1
    return output


# ------------------------------------------------
# ------------------------------------------------
# функция поиска патерна Bullish Engulfing
def CheckPatternBullishEngulfing(df_Open, df_Close, df_AvgBody, df_MdOC, df_CloseAvg):
    k = len(df_Open)
    output = np.zeros(k)
    i = 3
    while i < k:
        if ((df_Open[i - 1] > df_Close[i - 1]) and \
                ((df_Close[i] - df_Open[i]) > df_AvgBody[i]) and \
                (df_Close[i] > df_Open[i - 1]) and \
                (df_MdOC[i - 1] < df_CloseAvg[i - 1]) and \
                (df_Open[i] < df_Close[i - 1])):
            output[i] = 1
        i += 1
    return output


# ------------------------------------------------


# ------------------------------------------------
# функция поиска внутреннего дня
def CheckInnerDay(df_High, df_Low):
    k = len(df_Low)
    output = np.zeros(k)
    i = 2
    while i < k:
        if ((df_Low[i] > df_Low[i - 1]) and (df_High[i] < df_High[i - 1])):
            output[i] = 1
        i += 1
    return output


# -------------------------------

# ------------------------------------------------
# функция поиска внешнего дня
def CheckExternalDay(df_High, df_Low):
    k = len(df_Low)
    output = np.zeros(k)
    i = 2
    while i < k:
        if ((df_Low[i] < df_Low[i - 1]) and (df_High[i] > df_High[i - 1])):
            output[i] = 1
        i += 1
    return output


# -------------------------------

# ------------------------------------------------
# функция поиска краткосрочного минимума
def CheckShortTermMin(df_Low, df_InnerDay, df_ExternalDay):
    k = len(df_Low)  # определяем длину массива
    output = np.zeros(k)  # формируем нулевой вектор
    i = 2  # устанавливаем счетчик
    while i < k - 1:  # пока не конец массива
        if ((df_Low[i] < df_Low[i - 1]) and (df_Low[i] < df_Low[i + 1]) \
                and (df_InnerDay[i] == 0) and (df_ExternalDay[i] == 0)):  # и если это не внешний и внутренний день
            output[i] = 1  # устанавливаем признак если выполенено условие
        i += 1  # увеличиваем счетчик
    return output  # возвращаем массив


# ------------------------------------------------

# функция поиска краткосрочного максимума
def CheckShortTermMax(df_High, df_InnerDay, df_ExternalDay):
    k = len(df_High)  # определяем длину массива
    output = np.zeros(k)  # формируем нулевой вектор
    i = 2  # устанавливаем счетчик
    while i < k - 1:  # пока не конец массива
        if ((df_High[i] > df_High[i - 1]) and (df_High[i] > df_High[i + 1]) \
                and (df_InnerDay[i] == 0) and (df_ExternalDay[
                                                   i] == 0)):  # если предыдущий и последующий максимум болше текущего максимума и если это не внешний и внутренний день
            output[i] = 1  # устанавливаем признак если выполенено условие
        i += 1  # увеличиваем счетчик
    return output  # возвращаем массив


# ------------------------------------------------

# ------------------------------------------------
# функция поиска минимума
def CheckTermMin(df_TermMin, df_Low):
    k = len(df_TermMin)
    output = np.zeros(k)
    i = 2
    j_1 = 0
    j_2 = 0
    localMin_1 = 0
    localMin_2 = 0

    while i < k - 1:
        if (df_TermMin[i] == 1):
            if j_1 == 0:
                localMin_1 = df_Low[i]
                j_1 = i
            elif j_2 == 0:
                localMin_2 = df_Low[i]
                j_2 = i
            else:
                if ((localMin_1 > localMin_2) and (df_Low[i] > localMin_2)):
                    output[j_2] = 1
                j_1 = j_2
                j_2 = i
                localMin_1 = localMin_2
                localMin_2 = df_Low[i]
        i += 1
    return output


# ------------------------------------------------

# ------------------------------------------------
# функция поиска максимума
def CheckTermMax(df_TermMax, df_High):
    k = len(df_TermMax)
    output = np.zeros(k)
    i = 2
    j_1 = 0
    j_2 = 0
    localMax_1 = 0
    localMax_2 = 0

    while i < k - 1:
        if (df_TermMax[i] == 1):
            if j_1 == 0:
                localMax_1 = df_High[i]
                j_1 = i
            elif j_2 == 0:
                localMax_2 = df_High[i]
                j_2 = i
            else:
                if ((localMax_1 < localMax_2) and (df_High[i] < localMax_2)):
                    output[j_2] = 1
                j_1 = j_2
                j_2 = i
                localMax_1 = localMax_2
                localMax_2 = df_High[i]
        i += 1
    return output


# ------------------------------------------------

# ------------------------------------------------
# функция определение тренда
def CheckTrend(df_TermMin, df_TermMax):
    k = len(df_TermMax)
    output = np.zeros(k)
    i = 2
    trend = 0
    while i < k - 1:
        if ((df_TermMin[i] == 1) and (df_TermMax[i] == 1)):
            trend = 0
        elif (df_TermMax[i] == 1):
            trend = -1
        elif (df_TermMin[i] == 1):
            trend = 1
        output[i] = trend
        i += 1
    return output


# ------------------------------------------------


# ------------------------------------------------
# функция поиска патерна Звезда

def CheckPatternMyStar(df_Open, df_High, df_Low, df_Close, df_AvgBody, df_MdOC):
    k = len(df_Open)
    output = np.zeros(k)
    i = 3
    while i < k:
        if ((df_High[i] - df_Close[i] > 0.007) and \
                (abs(df_Open[i] - df_Close[i]) < 0.001) and \
                ((df_Open[i] - df_Low[i]) / (df_High[i] - df_Close[i]) < 0.20)):
            output[i] = 1
        i += 1
    return output


# ------------------------------------------------


# return np.average(series[-n:])



# подключаем библиотеку работы с графикой
# import plotly as py
# from plotly import tools

# import plotly.plotly as py
# import plotly.graph_objs as go
# from plotly.tools import FigureFactory as FF

# Задаем параметры загрузки
PeriodTiket = "'H4'"

# загружаем данные из файла
# создаем подключение
# con = pyodbc.connect('DRIVER={SQL Server};SERVER=LAPTOP-9AD7JQAO\SQLEXPRESS;DATABASE=Forex')
con = pyodbc.connect('DRIVER={SQL Server};Server=194.87.236.180;Database=ForexSQL;Uid=sysAndrey;Pwd=aikido;')

# заружаем данные
df_4h = pd.read_sql("select * from ArrayРredikator('EURUSD'," + PeriodTiket + ") order by Дата", con)

# альтернативная загрузка из файла
# df_f4 =pd.read_excel('C:\Gulyagin\Forex\ИсхДанные\ForexDate.xlsx','h4')
df = df_4h

# преобразовываем массив

df.columns = ['CurrencyPair', 'Период', 'Дата', 'Год', 'Месяц', 'День', 'Час', 'Минута', 'НеделяГода', 'ДеньНедели', \
              'OPEN', 'HIGH', 'LOW', 'CLOSE', 'TICKVOL', 'VOL', 'SPREAD', 'MED', 'TYP', 'WG', 'MdOC', \
              '#CO', '#HO', '#LO']

# добвляем производные цены и параметры
# скользящая средняя
df['MA'] = moving_average(df['CLOSE'], 24)
# скользящая средняя
df['Body'] = abs(df['OPEN'] - df['CLOSE']) * 1000
# среднее тело свечи
df['AvgBody'] = AvgBody(df['OPEN'], df['CLOSE'])
#
df['PtMorningStar'] = CheckPatternMorningStar(df['OPEN'], df['CLOSE'], df['AvgBody'], df['MdOC'])
##
df['PtEveningStar'] = CheckPatternEveningStar(df['OPEN'], df['CLOSE'], df['AvgBody'], df['MdOC'])
#
df['PtBearishEngulfing'] = CheckPatternBearishEngulfing(df['OPEN'], df['CLOSE'], df['AvgBody'], df['MdOC'], df['MA'])
#
df['PtBullishEngulfing'] = CheckPatternBullishEngulfing(df['OPEN'], df['CLOSE'], df['AvgBody'], df['MdOC'], df['MA'])
#
df['InnerDay'] = CheckInnerDay(df['HIGH'], df['LOW'])  # внутренний день
#
df['ExternalDay'] = CheckExternalDay(df['HIGH'], df['LOW'])  # внутренний день
#
df['ShortTermMin'] = CheckShortTermMin(df['LOW'], df['InnerDay'], df['ExternalDay'])  # краткосрочный минимум

df['ShortTermMax'] = CheckShortTermMax(df['HIGH'], df['InnerDay'], df['ExternalDay'])  # краткосрочный максимум
# поиск среднесрочных минимов
df['MediumTermMin'] = CheckTermMin(df['ShortTermMin'], df['LOW'])
# поиск среднесрочных максимумов
df['MediumTermMax'] = CheckTermMax(df['ShortTermMax'], df['HIGH'])
# поиск долгосрочных минимов
df['LongTermMin'] = CheckTermMin(df['MediumTermMin'], df['LOW'])
# поиск долгосрочных максимумов
df['LongTermMax'] = CheckTermMax(df['MediumTermMax'], df['HIGH'])
# определение долгосрочного тренда
df['ShortTrend'] = CheckTrend(df['ShortTermMin'], df['ShortTermMax'])
# определение долгосрочного тренда
df['MediumTrend'] = CheckTrend(df['MediumTermMin'], df['MediumTermMax'])
# определение долгосрочного тренда
df['LongTrend'] = CheckTrend(df['LongTermMin'], df['LongTermMax'])

df['MyStar'] = CheckPatternMyStar(df['OPEN'], df['HIGH'], df['LOW'], df['CLOSE'], df['AvgBody'], df['MdOC'])

# df = df.set_index(df.DATE)
# устанавливаем индекс


# df = df [['DATE','TIME','OPEN','HIGH','LOW','CLOSE','TICKVOL','VOL','SPREAD']]

# df = df [['OPEN','HIGH','LOW','CLOSE','MA']]


if not os.path.isdir(PathData):
    os.mkdir(PathData)


Filename = 'EURUSD.H4.xlsx'

FullFileName = PathData + '\\'+ Filename

df.to_excel(FullFileName, index=False)

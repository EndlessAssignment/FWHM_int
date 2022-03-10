from typing import Any
import pandas as pd
from glob import glob
from heapq import nsmallest
import numpy as np


# 반치폭, 반치폭의 광파워, 피크 파장 구하는 클래스
class Calc:
    def __init__(self, y_values_temp, x_values_temp):
        # array 형태의 x,y 값을 리스트화
        self.x_values, self.y_values, self.temp_l, self.temp_r = [], [], [], []
        self.y_values_temp = y_values_temp
        self.x_values_temp = x_values_temp
        for x in range(0, len(self.y_values_temp)):
            self.y_values.append(self.y_values_temp[x])
        for i in range(0, len(self.x_values_temp)):
            self.x_values.append(self.x_values_temp[i])

        # 피크와 피크의 절반
        self.peak_height = max(self.y_values)
        self.half_peak_height = max(self.y_values) / 2

        # 피크 파장을 기준으로 y 값을 둘로 나눈다
        self.y_r_temp = self.y_values[self.y_values.index(self.peak_height):len(self.y_values)]
        self.y_l_temp = self.y_values[0:self.y_values.index(self.peak_height)]

        # 피크 왼쪽과 오른쪽에서 각각 피크 절반에 가장 가까운 값을 찾는다
        self.y_r = nsmallest(1, self.y_r_temp, key=lambda a: abs(a - self.half_peak_height))
        self.y_l = nsmallest(1, self.y_l_temp, key=lambda b: abs(b - self.half_peak_height))

        # 아까 찾은 두 값에 대응하는 x 값을 찾는다
        self.temp_l.append(self.x_values[self.y_l_temp.index(self.y_l[0])])
        self.temp_r.append(self.x_values[self.y_r_temp.index(self.y_r[0]) + len(self.y_l_temp) - 1])

        # 반치폭 내에 해당하는 x 값과 y 값
        self.x_range = self.x_values[self.x_values.index(self.temp_l):self.x_values.index(self.temp_r)]
        self.y_range = self.y_values[self.y_values.index(self.y_l):self.y_values.index(self.y_r)-1]

        # 값을 담기 위한 초기 값
        self.sum = 0

    # 반치폭을 리턴하는 함수
    def fwhm(self):
        fwhm_n = self.temp_l[0] - self.temp_r[0]
        return abs(fwhm_n)

    # 반치폭의 광량을 적분해 리턴하는 함수, 구분구적법 사용
    def power(self):
        for n in range(0, len(self.x_range)):
            self.sum += (self.x_range[1] - self.x_range[0]) * self.y_range[n]
        return self.sum

    # 피크 파장을 리턴하는 함수
    def peak(self):
        return self.x_values[self.y_values.index(self.peak_height)]


# 불러온 데이터들 저장하는 함수
# noinspection PyArgumentList
def data_save(folder, peak, laser):
    data = []
    file = glob(folder + '/spec/*mw.xlsx', recursive=True)
    i: str | bytes | Any
    for i in file:
        try:
            df_temp = pd.read_excel(i)
            df = df_temp.drop(index=[0, 1, 2, 3, 4], axis=0)  # 측정 데이터가 없는 행들을 잘라냄
            x = list(df['Filename-->'])  # 파장
            y = list(df['Unnamed: 1'])  # 인텐시티

            # 피크 파장과 레이저 광 사이에서 최솟값을 찾는다
            p_approx = nsmallest(1, x, key=lambda c: abs(c-peak))[0]
            l_approx = nsmallest(1, x, key=lambda d: abs(d-laser))[0]
            y_temp = y[x.index(l_approx):x.index(p_approx)]
            point_temp = y_temp.index(min(y_temp))
            point = point_temp + x.index(l_approx)

            # 최소값 이후 값들만 LED 광량이라고 추측
            wavelength = np.array(x[point:])
            intensity = np.array(y[point:])
            time = df_temp['Unnamed: 1'][1]  # 적분시간
            power = float(i[i.index('\\')+1:i.index('mW')])  # LD 파워, 파일 이름에서 추출했음

            # 클래스를 이용해서 리스트화
            temp = Calc(intensity, wavelength)
            result = [power, temp.power() / time, temp.fwhm(), temp.peak(), 1240 / temp.peak()]
            data.append(result)

        except:
            # 오류 발생시 파워 출력
            print(i)

    # 데이터 프레임 화
    dt_array = np.array(data)
    names = ['Excitation Power (mW)', 'light Output Power (a.u.)', 'FWHM (nm)',
             'Peak Wavelength (nm)', 'Photon Energy (eV)']
    df_data = pd.DataFrame(dt_array, columns=names)

    # 파워에 대해 오름차순으로 정렬
    df_to_save = df_data.sort_values(by=df_data.columns[0])

    # 저장
    df_to_save.to_csv(path + '/data.csv', index=False)


# 파일 경로와 피크 파장, 적당히 잘 입력 바람
path = 'C:/Users/PJH/Desktop/연구실/실험데이터/PL/220308/100K'
data_save(path, 450, 405)

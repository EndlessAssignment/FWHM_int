import pandas as pd
from heapq import nsmallest
import numpy as np


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

    # 평균 파장을 리턴하는 함수
    def mean_wavelength(self):
        weight = 0
        for i in range(0, len(self.y_range)):
            wavelength = self.x_range[i]
            intensity = self.y_range[i]
            weight += wavelength * (self.x_range[1] - self.x_range[0]) * intensity
        return weight / self.sum


# 불러온 데이터들 저장하는 함수
def data_save(file):
    # 리스트 초기화, errors 는 에러가 발생한 전류를 담는 리스트
    data = []
    errors = []

    df = pd.read_csv(file)
    wavelength = df['Wavelength']  # 파장
    current_list = df.columns[1:]  # 전류
    for current in current_list:
        try:
            # 클래스를 이용해서 리스트화
            intensity = df[current]
            temp = Calc(intensity, wavelength)
            result = [current, temp.power(), temp.power()/float(current), temp.fwhm(), temp.peak(), 1240 / temp.peak(),
                      temp.mean_wavelength(), 1240 / temp.mean_wavelength()]
            data.append(result)
        except:
            # 에러 발생시 전류값을 리스트에 추가
            errors.append(current)

    # 데이터 프레임 화
    dt_array = np.array(data)
    names = ['Current (A)', 'light Output Power (a.u.)', 'EQE (a.u.)', 'FWHM (nm)',
             'Peak Wavelength (nm)', 'Peak Photon Energy (eV)', 'Mean Wavelength (nm)', 'Mean Photon Energy (eV)']
    df_data = pd.DataFrame(dt_array, columns=names)

    # 데이터 저장
    df_data.to_csv('C:/Users/PJH/Desktop/test.csv', index=False)
    print(errors)


data_save('C:/Users/PJH/Downloads/UV_LED_268_저온/220111_UV저온_268/268 spec/300K_spectrum_csv.csv')

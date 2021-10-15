import pandas as pd
from glob import glob
from heapq import nsmallest
import numpy as np
from collections import OrderedDict


# 반치폭, 반치폭의 광파워, 피크 파장 구하는 클래스
class calc:
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
        self.y_r = nsmallest(1, self.y_r_temp, key=lambda x: abs(x - self.half_peak_height))
        self.y_l = nsmallest(1, self.y_l_temp, key=lambda x: abs(x - self.half_peak_height))

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
            self.sum += self.x_range[n] * self.y_range[n]
        return self.sum

    # 피크 파장을 리턴하는 함수
    def peak(self):
        return self.x_values[self.y_values.index(self.peak_height)]


# 파일 경로, 적당히 잘 입력 바람
file = glob('C:/Users/PJH/Desktop/연구실/실험데이터/211002/400 K/int_cons/*mW.xlsx', recursive=True)

data = {}
for i in file:
    try:
        df_temp = pd.read_excel(i)
        df = df_temp.drop(index=[0, 1, 2, 3, 4], axis=0)  # 측정 데이터가 없는 행들을 잘라냄
        x = list(df['Filename-->'])  # 파장
        y = list(df['Unnamed: 1'])  # 인텐시티
        point = y.index(min(y[40:350]))  # 적당한 구간에서 최소값을 찾는다, 구간은 상황에 따라 조절바람

        # 최소값 이후 값들만 LED 광량이라고 추측
        wavelength = np.array(x[point:])
        intensity = np.array(y[point:])
        time = df_temp['Unnamed: 1'][1]  # 적분시간
        power = float(i[i.index('\\')+1:i.index('mW')])  # LD 파워, 파일 이름에서 추출했음

        # 클래스를 이용해서 딕셔너리화
        temp = calc(intensity, wavelength)
        data[power] = [temp.power() / time, temp.fwhm(), temp.peak()]
    except:
        print(i)

# 파워 오름차순으로 정렬함
arrange = sorted(data.items())

# 데이터를 담기 위한 초기 값
input_power = []
output_power = []
full_half = []
peak_wavelength = []

# 정렬한 것을 하나씩 집어 넣는다, 이 과정이 굉장히 더러운데 최적화 시킬 수 있으면 해주기 바람
for i in arrange:
    input_power.append(list(i)[0])
    result = list(list(i)[1])
    output_power.append(result[0])
    full_half.append(result[1])
    peak_wavelength.append(result[2])

# 데이터 프레임화 시키기 위한 과정, 사실 잘 모르겠음
final = OrderedDict(
    [
        ('Excitation Power (mW)', input_power),
        ('light Output Power (a.u.)', output_power),
        ('FWHM (nm)', full_half),
        ('Peak Wavelength (nm)', peak_wavelength)
    ]
)

# 데이터 저장
sex = pd.DataFrame.from_dict(final).set_index('Excitation Power (mW)')
sex.to_csv('data.csv')

import numpy as np
import pandas as pd
import struct
import tensorflow as tf

class Acustic_Sample():
    def __init__(self, path, modelpath, step=0.1): #Задать в path путь к файлу а в modelpath путь к модели
        self.path=path
        self.step=step
        self.model=tf.keras.models.load_model(modelpath)
        dat_1 = []
        dat_2 = []
        dat_3 = []
        dat_4 = []
        dat_5 = []
        dat_6 = []
        dat_7 = []

        self.file_open()
        self.get_data(dat_1, 0)
        self.get_data(dat_2, 1)
        self.get_data(dat_3, 2)
        self.get_data(dat_4, 3)
        self.get_data(dat_5, 4)
        self.get_data(dat_6, 5)
        self.get_data(dat_7, 6)
        time = []


        for i in range(self.n_otsch[0]):
            time.append(i * 5 / 3 / 1000)

        self.time=time
        self.dat_1=dat_1
        self.dat_2=dat_2
        self.dat_3=dat_3
        self.dat_4=dat_4
        self.dat_5=dat_5
        self.dat_6=dat_6
        self.dat_7=dat_7
        self.preprocessing()

#Парсим ук-файл
    def file_open(self):
        with open(self.path, "rb") as f:
            self.chan = struct.unpack('H', f.read(2))
            self.n_otsch = struct.unpack('H', f.read(2))
            self.num_1 = struct.unpack('I', f.read(4))
            self.discr = struct.unpack('d', f.read(8))
            self.num_3 = struct.unpack('d', f.read(8))
            self.num_4 = struct.unpack('h', f.read(2))
            self.num_5 = struct.unpack('I', f.read(4))
            self.num_6 = struct.unpack('I', f.read(4))
            f.seek(256)
            bug = f.read()
            f.seek(256)
            self.lst = []
            num = 0
            while num < self.chan[0] * self.n_otsch[0]:
                num += 1
                self.lst.append(struct.unpack('d', f.read(8)))
#Функция для получения отсчетов по отдельному каналу
    def get_data(self, data, Nchan):
        for i in range(self.n_otsch[0]):
            data.append(self.lst[i + int(self.n_otsch[0]) * Nchan][0])
        return data

#Preprocessing data
    def preprocessing(self):
        #AbsMaxScaling data
        mat_raw = self.dframe_wotime().T.values
        mat = mat_raw/np.max(np.abs(mat_raw))
        mat = np.expand_dims(mat, 2)
        mat = np.expand_dims(mat, 0)

        #StepImplimentation
        lenght = np.array([k * self.step for k in range(mat_raw.shape[0])])
        lenght = np.expand_dims(lenght, 0)

        #Spectr FFT
        mat_spectr = np.abs(np.fft.fft(mat_raw,axis=0))
        mat_spectr = (mat_spectr/mat_spectr.max()).T
        mat_spectr = np.expand_dims(mat_spectr, 0)
        return [mat,mat_spectr,lenght]

#Вывод скоростей
    def velocity(self):
        vel =self.model.predict(self.preprocessing())
        return vel

#PD frame
    def dframe(self):
        df = pd.DataFrame({'t':self.time,'1':self.dat_1,'2':self.dat_2,'3':self.dat_3,'4':self.dat_4,'5':self.dat_5,'6':self.dat_6,'7':self.dat_7,})
        return df
    def dframe_wotime(self):
        df = pd.DataFrame({'1':self.dat_1,'2':self.dat_2,'3':self.dat_3,'4':self.dat_4,'5':self.dat_5,'6':self.dat_6,'7':self.dat_7,})
        return df

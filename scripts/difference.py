import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, lfilter
from math import cos, radians

class KalmanFilter(object):
    def __init__(self, F = None, B = None, H = None, Q = None, R = None, P = None, x0 = None):

        if(F is None or H is None):
            raise ValueError("Set proper system dynamics.")

        self.n = F.shape[1]
        self.m = H.shape[1]

        self.F = F
        self.H = H
        self.B = 0 if B is None else B
        self.Q = np.eye(self.n) if Q is None else Q
        self.R = np.eye(self.n) if R is None else R
        self.P = np.eye(self.n) if P is None else P
        self.x = np.zeros((self.n, 1)) if x0 is None else x0

    def predict(self, u = 0):
        self.x = np.dot(self.F, self.x) + np.dot(self.B, u)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return self.x

    def update(self, z):
        y = z - np.dot(self.H, self.x)
        S = self.R + np.dot(self.H, np.dot(self.P, self.H.T))
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x = self.x + np.dot(K, y)
        I = np.eye(self.n)
        self.P = np.dot(np.dot(I - np.dot(K, self.H), self.P), 
        	(I - np.dot(K, self.H)).T) + np.dot(np.dot(K, self.R), K.T)
        

def lat_lon_diff_in_meters(lat1, lon1, lat2, lon2):
    # Convert degrees to radians
    lat1 = radians(lat1)
    lon1 = radians(lon1)
    lat2 = radians(lat2)
    lon2 = radians(lon2)

    # Differences in coordinates
    dlat = (lat2 - lat1) * 111200
    dlon = (lon2 - lon1) * 111200 * cos((lat1+lat2)/2)

    return dlat, dlon

def floating_average(arr, window_size):
    averages = []
    for i in range(len(arr)):
        if i < window_size:
            average = sum(arr[:i+1]) / (i+1)
        else:
            average = sum(arr[i-window_size+1:i+1]) / window_size
        averages.append(average)
    return averages

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def plot_3d(x, y, z):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot3D(x, y, z, 'red')

    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_zlabel('z (m)')

    plt.show()

def plot_2d(x, y):
    plt.plot(x, y)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

def main():
    gps = pd.read_csv('~/Desktop/Uni/5.Semester/Drohnen-Projekt/IMU/data/standing/gps.csv', sep=', ')
    lat = gps['gps_lat']
    lon = gps['gps_long']
    alt = gps['gps_alt']

    sensor = pd.read_csv('~/Desktop/Uni/5.Semester/Drohnen-Projekt/IMU/data/flyOut/combined.csv', sep=', ')

    time = sensor['timestamp']
    accX = sensor['acc0']
    accY = sensor['acc1']
    accZ = sensor['acc2']
    yaw = sensor['gyro_rad0']
    pitch = sensor['gyro_rad1']
    roll = sensor['gyro_rad2']

    test_lat = []
    test_lon = []
    test_alt = []

    for i in range(len(lat)):
        test_lat.append(lat_lon_diff_in_meters(lat[0], lon[0], lat[i], lon[i])[0])
        test_lon.append(lat_lon_diff_in_meters(lat[0], lon[0], lat[i], lon[i])[1])
        test_alt.append(alt[i]- alt[0])

    avg_accZ = np.mean(accZ[:10])
    accZ -= avg_accZ

    dt = (time[1] - time[0]) / 1_000_000_000
    F = np.array(
        [[1, dt, 0.5*dt**2], 
         [0, 1, dt], 
         [0, 0, 1]
        ])
    H = np.array([1, 0, 0]).reshape(1, 3)
    Q = np.array(
        [[0, 0.1, 0.2], 
         [0.0, 0.1, 0.2], 
         [0.0, 0.0, 0.1]
        ])
    R = np.array([200]).reshape(1, 1)

    Q2 = np.array(
        [[0, 175, 200], 
        [0.0, 175, 200], 
        [0.0, 0.0, 150]
        ])

    R2 = np.array([150]).reshape(1, 1)


    radKf = KalmanFilter(F = F, H = H, Q = Q, R = R)
    accKf = KalmanFilter(F = F, H = H, Q = Q2, R = R2)

    messure = roll
    filter = radKf

    predictionsX = []
    predictionsY = []
    predictionsZ = []
    

    for x in messure:
        predictionsX.append(np.dot(H,  filter.predict())[0])
        filter.update(x)
    
    plt.plot(range(len(messure)), messure, label = 'pitch')  
    plt.plot(range(len(predictionsX)), np.array(predictionsX), label = 'Kalman Filter Prediction')
    plt.legend()
    plt.show()
    


    '''
    test_lat = []
    test_lon = []
    test_alt = []

    for i in range(len(lat)):
        test_lat.append(lat_lon_diff_in_meters(lat[0], lon[0], lat[i], lon[i])[0])
        test_lon.append(lat_lon_diff_in_meters(lat[0], lon[0], lat[i], lon[i])[1])
        test_alt.append(alt[i]- alt[0])
    
    
    test_lat = butter_lowpass_filter(test_lat, 3, 150)
    test_lon = butter_lowpass_filter(test_lon, 3, 150)
    test_alt = floating_average(test_alt, 20)
    '''

    #plot_3d(test_lat, test_lon, test_alt)

    

if __name__ == '__main__':
    main()

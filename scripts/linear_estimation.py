import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

def plot_3d(x, y, z):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot3D(x, y, z, 'red')

    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_zlabel('z (m)')

    plt.show()

def plot_2d(real, predicted):
    plt.plot(range(len(real)), real, label = 'real')  
    plt.plot(range(len(predicted)), np.array(predicted), label = 'Kalman Filter Prediction')
    plt.legend()
    plt.show()

def plot_diff(diff):
    plt.figure()
    plt.plot(diff)
    plt.title('Spatial Difference Between GPS and calculated position')
    plt.xlabel('Index')
    plt.ylabel('Euclidean Distance')
    plt.show()

def calc_difference(ax, ay, az, bx, by, bz):
    diff = []
    for i in range(len(ax)):
        diff.append(np.sqrt((bx[i] - ax[i])**2 + (by[i] - ay[i])**2 + (bz[i] - az[i])**2))
    return diff
        

def floating_average(arr, window_size):
    averages = []
    for i in range(len(arr)):
        if i < window_size:
            average = sum(arr[:i+1]) / (i+1)
        else:
            average = sum(arr[i-window_size+1:i+1]) / window_size
        averages.append(average)
    return averages

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

def main():
    #input data
    time = []
    accX, accY, accZ = [], [], []

    #drone calculated data
    posX, posY, posZ = [], [], []

    #calculated data
    velX, velY, velZ = 0, 0, 0

    #output data
    calcX, calcY, calcZ = [], [], []

    path = "hover"

    gps = pd.read_csv(f'~/Desktop/Uni/5.Semester/Drohnen-Projekt/IMU/data/{path}/gps.csv', sep=', ')
    lat = gps['gps_lat']
    lon = gps['gps_long']
    alt = gps['gps_alt']

    test_lat = []
    test_lon = []
    test_alt = []

    for i in range(len(lat)):
        test_lat.append(lat_lon_diff_in_meters(lat[0], lon[0], lat[i], lon[i])[0])
        test_lon.append(lat_lon_diff_in_meters(lat[0], lon[0], lat[i], lon[i])[1])
        test_alt.append(alt[i]- alt[0])

    
    sensor = pd.read_csv(f'~/Desktop/Uni/5.Semester/Drohnen-Projekt/IMU/data/{path}/combined.csv', sep=', ')
    time = sensor['timestamp']
    accX = sensor['acc0']
    accY = sensor['acc1']
    accZ = sensor['acc2']

    avg_accZ = np.average(accZ)
    print(avg_accZ)
    accZ -= avg_accZ

    dt = (time[1] - time[0]) / 1_000_000_000
    F = np.array(
        [[1, dt, 0.5*dt**2], 
         [0, 1, dt], 
         [0, 0, 1]
        ])
    H = np.array([1, 0, 0]).reshape(1, 3)
    Q = np.array(
        [[0, 175, 300], 
         [0, 175, 200], 
         [0, 0, 150]
        ])
    R = np.array([200]).reshape(1, 1)

    filterX = KalmanFilter(F=F, H=H, Q=Q, R=R)
    filterY = KalmanFilter(F=F, H=H, Q=Q, R=R)
    filterZ = KalmanFilter(F=F, H=H, Q=Q, R=R)

    filteredX = []
    filteredY = [] 
    filteredZ = []
    
    for x in accX:
        filteredX.append(np.dot(H,  filterX.predict())[0])
        filterX.update(x)
    
    for y in accY:
        filteredY.append(np.dot(H,  filterY.predict())[0])
        filterY.update(y)

    for z in accZ:
        filteredZ.append(np.dot(H,  filterZ.predict())[0])
        filterZ.update(z)

    print(filteredZ[0])
    for i in range(len(time)-1):
        dt = (time[i+1] - time[i]) / 1_000_000

        # Integrate acceleration to get velocity
        velX += filteredX[i] * dt
        velY += filteredY[i] * dt
        velZ -= filteredZ[i] * dt

        # Integrate velocity to get position
        calcX.append(velX * dt)
        calcY.append(velY * dt)
        calcZ.append(velZ * dt)
        
        if i != 0:
            calcX[i] += calcX[i-1]
            calcY[i] += calcY[i-1]
            calcZ[i] += calcZ[i-1]
    
    print(calcZ[0])
    plot_2d(accZ, filteredZ)
    plot_3d(calcX, calcY, calcZ)

if __name__ == "__main__":
    main()

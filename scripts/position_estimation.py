import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

def xRot(theta):
    return np.array(
        [[1, 0, 0, 0], 
         [0, np.cos(theta), -np.sin(theta), 0], 
         [0, np.sin(theta), np.cos(theta), 0],
         [0, 0, 0, 1]
    ])

def yRot(theta):
    return np.array(
        [[np.cos(theta), 0, np.sin(theta), 0],
         [0, 1, 0, 0],
         [-np.sin(theta), 0, np.cos(theta), 0],
         [0, 0, 0, 1]
        ])

def zRot(theta):
    return np.array(
        [[np.cos(theta), -np.sin(theta), 0, 0],
         [np.sin(theta), np.cos(theta), 0, 0],
         [0, 0, 1, 0],
         [0, 0, 0, 1]
        ])

def trans(x, y, z):
    return np.array(
        [[1, 0, 0, x],
         [0, 1, 0, y],
         [0, 0, 1, z],
         [0, 0, 0, 1]
        ])

def plot_2d(real, predicted):
    plt.plot(range(len(real)), real, label = 'real')  
    plt.plot(range(len(predicted)), np.array(predicted), label = 'Kalman Filter Prediction')
    plt.legend()
    plt.show()

def plot_3d(x, y, z):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot3D(x, y, z, 'red')

    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_zlabel('z (m)')

    plt.show()

def plot_diff(diff):
    plt.figure()
    plt.plot(diff)
    plt.title('Spatial Difference Between pos and calc')
    plt.xlabel('Index')
    plt.ylabel('Euclidean Distance')
    plt.show()

def calc_difference(first, second):
    return [np.linalg.norm(np.array(a) - np.array(b)) for a, b in zip(first, second)]

def floating_average(arr, window_size):
    averages = []
    for i in range(len(arr)):
        if i < window_size:
            average = sum(arr[:i+1]) / (i+1)
        else:
            average = sum(arr[i-window_size+1:i+1]) / window_size
        averages.append(average)
    return averages

def main():
    #input data
    time = []
    accX, accY, accZ = [], [], []
    yaw, pitch, roll = [], [], []

    #drone calculated data
    posX, posY, posZ = [], [], []

    #calculated data
    velX, velY, velZ = 0, 0, 0

    #output data
    calcX, calcY, calcZ = [], [], []
    
    sensor = pd.read_csv('~/Desktop/Uni/5.Semester/Drohnen-Projekt/IMU/data/circle/combined.csv', sep=', ')
    time = sensor['timestamp']
    accX = sensor['acc0']
    accY = sensor['acc1']
    accZ = sensor['acc2']
    yaw = sensor['gyro_rad0']
    pitch = sensor['gyro_rad1']
    roll = sensor['gyro_rad2']

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
    Q2 = np.array(
        [[0, 0.1, 0.2], 
         [0.0, 0.1, 0.2], 
         [0.0, 0.0, 0.1]
        ])
    R2 = np.array([200]).reshape(1, 1)

    filterX = KalmanFilter(F=F, H=H, Q=Q, R=R)
    filterY = KalmanFilter(F=F, H=H, Q=Q, R=R)
    filterZ = KalmanFilter(F=F, H=H, Q=Q, R=R, x0 = np.array([accZ[0], 0, 0]))
    filterYaw = KalmanFilter(F=F, H=H, Q=Q2, R=R2)
    filterPitch = KalmanFilter(F=F, H=H, Q=Q2, R=R2)
    filterRoll = KalmanFilter(F=F, H=H, Q=Q2, R=R2)


    filteredX = []
    filteredY = [] 
    filteredZ = []
    filteredYaw = []
    filertedPitch = []
    filteredRoll = []
    
    for x in accX:
        filteredX.append(np.dot(H,  filterX.predict())[0])
        filterX.update(x)
    
    for y in accY:
        filteredY.append(np.dot(H,  filterY.predict())[0])
        filterY.update(y)

    for z in accZ:
        filteredZ.append(np.dot(H,  filterZ.predict())[0])
        filterZ.update(z)

    for y in yaw:
        filteredYaw.append(np.dot(H,  filterYaw.predict())[0])
        filterYaw.update(y)

    for p in pitch:
        filertedPitch.append(np.dot(H,  filterPitch.predict())[0])
        filterPitch.update(p)
    
    for r in roll:  
        filteredRoll.append(np.dot(H,  filterRoll.predict())[0])
        filterRoll.update(r)

    plot_2d(yaw, filteredYaw)

    gravity = 9.8
    rot = np.eye(4)
    for i in range(len(time)-1):

        dt = (time[i +1] - time[i]) / 1_000_000
        print(dt)
        
        grav = np.array([0, 0, gravity, 1])
        rot = rot @ zRot(yaw[i]) @ yRot(pitch[i]) @ xRot(roll[i])
        gravTurned = rot @ grav
        acc = (np.array([filteredX[i][0], filteredY[i][0], filteredZ[i], 1]) + gravTurned) * dt
        acc = trans(acc[0], acc[1], acc[2])

        accTurned = rot @ acc

        velX += accTurned[0, 3]
        velY += accTurned[1, 3]
        velZ -= accTurned[2, 3]

        # Integrate velocity to get position
        calcX.append(velX * dt)
        calcY.append(velY * dt)
        calcZ.append(velZ * dt)

        if i != 0 :
            calcX[i] += calcX[i-1]
            calcY[i] += calcY[i-1]
            calcZ[i] += calcZ[i-1]

    plot_3d(calcX, calcY, calcZ)

if __name__ == "__main__":
    main()
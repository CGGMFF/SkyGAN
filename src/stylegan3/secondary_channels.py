
'''
    init(RES=128) ... call this to change the resolution if 128x128 is not desired

    distance_to_clouds
        RESxRES array ... a stereographic-projected distance to a plane at 5000 m (approximating clouds position), clipped
    
    polar_distance
        phi ... azimuth: [0..2pi]
        theta ... elevation: [-pi/2..pi/2], positive === Sun is above horizon
        computes an Euclidean(?) distance between the two given directions
        returns: RESxRES array with values from [-1, 1]
'''


import numpy as np

class SecondaryChannels:
    @staticmethod
    def img_to_direction(x, y):
        result = np.zeros(x.shape+(3,))
        denom = (1+x*x+y*y)
        result[...,0] = 2*x/denom
        result[...,1] = 2*y/denom
        result[...,2] = (1-(x*x+y*y))/denom
        
        mask = (x*x + y*y) > 1.0
        result[mask,:] = 0.0
        return result

    @staticmethod
    def polar_to_direction(phi, theta, r = 1.0):
        theta = np.pi/2 - theta
        result = np.zeros(phi.shape+(3,))
        result[...,0] = r*np.cos(phi)*np.sin(theta)
        result[...,1] = r*np.sin(phi)*np.sin(theta)
        result[...,2] = r*np.cos(theta)
        return result

    @staticmethod
    def direction_to_polar(direction):
        theta = np.pi/2 - np.arccos(direction[...,2]/1.0)
        phi = np.arctan(direction[...,1] / direction[...,0])
        phi[direction[...,0] < 0] += np.pi
        return phi, theta

    @staticmethod
    def polar_distance(phi1, theta1, phi2, theta2):
        theta1 = np.pi/2 - theta1
        theta2 = np.pi/2 - theta2
        img = np.sqrt(1+1-2*1*1*(np.sin(theta1)*np.sin(theta2)*np.cos(phi2-phi1) + np.cos(theta1)*np.cos(theta2)))
        img = np.nan_to_num(img, nan=2)
        return (img - 1).astype(np.float32) # values from [-1,1]


    def __init__(self, resolution):
        print('secondary_channels init', resolution)
        self.resolution = resolution

        x_coords = np.linspace(-1., 1., resolution)
        y_coords = np.linspace(-1., 1., resolution)
        x, y = np.meshgrid(x_coords, y_coords)

        self.direction = self.img_to_direction(x, y)

        print('direction.shape', self.direction.shape)

        point = (0, 0, 5000) # "distance to clouds [m]" ... distance to a plane at height 5000 m
        self.distance_to_clouds = point[2] / self.direction[..., 2]
        self.distance_to_clouds = np.minimum(self.distance_to_clouds, 252597) # limit the values to avoid +Inf; use the "tangential" distance between two circles: Earth ~6378 km and clouds (Earth + 5km) ... sqrt((6378+5)^2-6378^2)
        # maybe we could ditch the planar clouds approximation for extra accuracy, but near the center it is good enough for now


        self.phi, self.theta = self.direction_to_polar(self.direction)

        sun = np.array((1.5*np.pi/2, np.arcsin(60/90)))
        sun_direction = self.polar_to_direction(sun[0], sun[1])
        sun, sun_direction, np.linalg.norm(sun_direction)

        #polar_distance(phi, theta, sun_phi, sun_theta)-1 # distribute values from [-1,1]
        

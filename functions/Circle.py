# import numpy library
import numpy as np

class Circle:
    """
    A class that implements a circle
    """
    # initialization requires center [x, y]
    # and radius of circle    
    def __init__(self, center, radius):
        self.center = center
        self.radius = radius
    
    # other derived attributes
    
    # circumference
    def circumference(self):
        return 2 * np.pi * self.radius
    
    # area
    def area(self):
        return np.pi * self.radius ** 2
    
    # x and y coordinates defining arc
    # default is full circle
    def coordinates(self, theta_start=0, theta_end=360):
        theta = np.arange(theta_start,theta_end) * np.pi / 180
        x = self.radius * np.cos(theta) + self.center[0]
        y = self.radius * np.sin(theta) + self.center[1]
        return x, y
    
    # x and y coordinates of radius at angle theta_end
    def radius_at_angle(self, theta_end):
        theta = theta_end * np.pi / 180
        x = np.array([0, self.radius * np.cos(theta) + self.center[0]])
        y = np.array([0, self.radius * np.sin(theta) + self.center[1]])
        return x, y
    
    # x and y coordinates of segment inside circle
    # defined by angles theta_start and theta_end
    def segment(self, theta_start, theta_end):
        theta = np.array([theta_start, theta_end]) * np.pi / 180
        x = self.radius * np.cos(theta) + self.center[0]
        y = self.radius * np.sin(theta) + self.center[1]
        return x, y
    
    # find the intersection of a line and the circle
    # line defined by y = m * x + c
    # code from GeeksforGeeks
    def find_intersections(self, m = 0.0, c = 0.0):
        h = self.center[0]
        k = self.center[1]
        r = self.radius

        A = 1 + m**2
        B = 2 * (m * (c - k) - h)
        C = h**2 + (c - k)**2 - r**2

        discriminant = B**2 - 4 * A * C

        # no intersection
        if discriminant < 0:
            return []
        # tangent
        elif discriminant == 0:
            x = -B / (2 * A)
            y = m * x + c
            return x, y
        # two intersections
        else:
            sqrt_discriminant = np.sqrt(discriminant)
            x1 = (-B + sqrt_discriminant) / (2 * A)
            x2 = (-B - sqrt_discriminant) / (2 * A)
            y1 = m * x1 + c
            y2 = m * x2 + c
            x = np.array([x1, x2])
            y = np.array([y1, y2])
            return x, y
    
    # methods
    
    # shift center in x
    def shift_in_x(self, x_value):
        self.center[0] += x_value
    
    # shift center in y
    def shift_in_y(self, y_value):
        self.center[1] += y_value
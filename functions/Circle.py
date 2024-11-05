# import numpy library
import numpy as np

class Circle:
    """
    A class that implements a circle,
    with methods that are useful for
    the Mohr's Circle
    """

    def __init__(self, center, radius):
        '''
        Initialize the circle with a center and radius
        '''
        self.center = center
        self.radius = radius
    
    # other derived attributes
    
    def circumference(self):
        '''
        Circumference of the circle
        '''
        return 2 * np.pi * self.radius
    
    def area(self):
        '''
        Area of the circle
        '''
        return np.pi * self.radius ** 2
    
    def coordinates(self, angle_start=0, angle_end=360):
        '''
        x and y coordinates defining arc of circle.
        Default is full circle
        '''
        theta = np.arange(angle_start,angle_end) * np.pi / 180
        x = self.radius * np.cos(theta) + self.center[0]
        y = self.radius * np.sin(theta) + self.center[1]
        return x, y
    
    def coordinates_at_angle(self, angle):
        '''
        x and y coordinates of radius at a given angle
        '''
        theta = angle * np.pi / 180
        x = self.radius * np.cos(theta) + self.center[0]
        y = self.radius * np.sin(theta) + self.center[1]
        
        return x, y
    
    # Methods

    def y_at_x(self, x):
        '''
        Get the y-coordinate of the circle at a given x-coordinate
        '''
        # Unpack the center coordinates
        h, k = self.center
        # Calculate the y-coordinate
        y = np.sqrt(self.radius ** 2 - (x - h) ** 2) + k
        return y
    
    def line_intersections(self, m, c):
        '''
        Given a line defined by y = mx + c
        Returns the intersections of the line and the circle
        '''
        # unpack the center coordinates
        h, k = self.center
        r = self.radius
        # calculate the intersection points
        A = 1 + m**2 # coefficient of x^2
        B = 2 * (m * c - m * k - h) # coefficient of x
        C = h**2 + (c - k)**2 - r**2 # constant term
        D = B**2 - 4 * A * C # discriminant
        if D < 0:
            return []
        elif D == 0:
            x = -B / (2 * A)
            y = m * x + c
            return [(x, y)]
        else:
            x1 = (-B + np.sqrt(D)) / (2 * A)
            y1 = m * x1 + c
            x2 = (-B - np.sqrt(D)) / (2 * A)
            y2 = m * x2 + c
            return [(x1, y1), (x2, y2)]         
    
    def chord_endpoint(self, point, angle):
        '''
        Find the endpoint of a chord on the circle
        given a point on the circle and the angle of
        the chord with respect to the x-axis
        '''
        # unpack center and starting point coordinates
        r = self.radius
        cx, cy = self.center
        x1, y1 = point
        # convert the angle to radians
        theta = np.radians(angle)
        # calculate the other endpoint of the chord using the angle 
        # from the x-axis
        x2 = x1 + 2 * r * np.cos(theta)
        y2 = y1 + 2 * r * np.sin(theta)
        # x2 and y2 must be on the circle
        if angle == 90 or angle == 270:
            y2 = self.y_at_x(x2)
        else:
            # slope and intercept of the line
            m = np.tan(theta)
            c = y1 - m * x1
            # intersection points of the line and circle
            intersections = self.line_intersections(m, c)
            if len(intersections) > 0:
                # choose the end point
                if np.allclose(intersections[0], [x1, y1]):
                    x2, y2 = intersections[1]
                else:
                    x2, y2 = intersections[0]
            else:
                x2, y2 = x1, y1 

        return x2, y2
    
    def fit_three_points(self, points):
        '''
        Modify the circle to fit three points
        The points are defined by a 3 x 2 numpy array
        '''
        # get the coordinates of the points
        x1, y1 = points[0]
        x2, y2 = points[1]
        x3, y3 = points[2]
        # set up the system of equations based on the 
        # expanded circle equation
        # x^2 + y^2 + Dx + Ey + F = 0
        A = np.array([
            [x1, y1, 1],
            [x2, y2, 1],
            [x3, y3, 1]
        ])
    
        B = np.array([
            [-(x1**2 + y1**2)],
            [-(x2**2 + y2**2)],
            [-(x3**2 + y3**2)]
        ])
        # solve for D, E, F using linear algebra
        D, E, F = np.linalg.solve(A, B).flatten()
        # Calculate the center (h, k) and radius r of the circle
        h = -D / 2
        k = -E / 2
        r = np.sqrt(h**2 + k**2 - F)
        # update the center and radius of the circle
        self.center = [h, k]
        self.radius = r
    
    def tangent_from_origin(self):
        '''
        Find the tangent line to the circle from the origin
        The center of the circle is at y = 0
        Returns the slope of the tangent and the x and y
        coordinates of the point of tangency
        '''
        h = self.center[0]
        r = self.radius
        # distance from the origin to the tangent point
        origin_to_tangent = np.sqrt(h**2 - r**2)
        # angle of radius to the tangent and the x-axis
        theta = np.arctan(origin_to_tangent / r)
        # x-coordinate of point of tangency
        x_tangent = h - r * np.cos(theta)
        # y-coordinate of point of tangency
        y_tangent = r * np.sin(theta)
        # slope of the tangent 
        slope = y_tangent / x_tangent
    
        return slope, x_tangent, y_tangent
         
    def shift_in_x(self, x_value):
        '''
        Shift the center of the circle in x
        '''
        self.center[0] += x_value
    
    def shift_in_y(self, y_value):
        '''
        Shift the center of the circle in y
        '''
        self.center[1] += y_value
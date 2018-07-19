"""
Script to simulate the spatial variation of a temperature process.

1. The entire area starts with the same temperature (effect of athmosphere).
2. There are sources of heat (e.g houses, traffic, people), and sources of 
cold (e.g. shadows, rivers, etc).
3. Each temperature source has a dissipated effect \Delta(d) = Src * 1/d^2, 
up to a maximum of d_max.

Inputs:
    - area width
    - coordinates of sensors in the area
    - grid of interest points.

Outputs:
The script generates a matrix of Mean Absolute Error (MAE) for every pair
of sensor, interest point. This matrix will be used to determine the error of
predicting the temperature value at an interest point from a sensor.
"""

d_max = 500*2

from random import random, seed
from math import sqrt, log
class TempSource():
    def __init__(self, delta, x, y):
        """
        Temperature source.

        Generates delta temperature at coordinates x,y.
        """
        self.delta = delta
        self.x = x
        self.y = y

class TemperatureProcess():

    def __init__(self, area_width, num_sources=100):
        """
        The spatial behaviour of a temperature process over an area.
        It has multiple sources of temperature (cold/heat), and the 
        effect is combined in measurement points.

        Parameters:
        area_width      -- width of the area
        num_sources     -- the number of temp sources (default 100)
        """
        self.area_width = area_width
        self.athmosphere = 15 + 10*random()
        self.num_sources = num_sources
        # Init the PRNG
        seed()  
        self.temp_sources = [
                TempSource(-5+10*random(), area_width*random(), area_width*random())
                for i in xrange(num_sources)]

    def temperature_process_mae(self, sensors, interest_points):
        """
        Determines the temperature at measurement points and interest
        points, and computes the mean absolute error (MAE) matrix
        between the sensors and interest points.

        Params:
        sensors         -- array of (x,y) coordinates
        interest_points -- array of (x,y) coordinates

        Return:
        MAE matrix, where MAE(ip_i, sp_j) = E[ip_i - sp_j]


        Keeps track of heat and cold sources separately.
        Only the most intense source has effect - so if there are
        two heat sources of +30 and +20, the overall impact will be +30.
        """
        sensor_temps = [[0,0] for n in sensors]
        interest_temps = [[0,0] for n in interest_points]
        for t in self.temp_sources:
            # Apply the impact of the temperature source to each sensor
            for idx, (x,y) in enumerate(sensors):
                dist = sqrt((t.x-x)**2 + (t.y-y)**2)
                impact = 0
                if dist <= 1:
                    impact = t.delta
                elif dist < d_max:
                    # Delta t is much smaller than distance: (-5,5) vs (0,500),
                    # so we use log to reduce the ratio.
                    # Also, the +1 is for avoiding higher impact than the delta
                    impact = t.delta/(1+log(10, dist))
                # Only the most intense sources have effect
                if impact < 0:
                    if sensor_temps[idx][0] > impact:
                        sensor_temps[idx][0] = impact
                if impact > 0:
                    if sensor_temps[idx][1] < impact:
                        sensor_temps[idx][1] = impact
            # Apply the impact of temperature at each interest point
            for idx, (x,y) in enumerate(interest_points):
                dist = sqrt((t.x-x)**2 + (t.y-y)**2)
                impact = 0
                if dist <= 1:
                    impact = t.delta
                elif dist < d_max:
                    impact = t.delta/(1+log(10, dist))
                if impact > 0:
                    pass
                    # print "Changing IP", idx, "by", impact
                # Only the most intense sources have effect
                if impact < 0:
                    if interest_temps[idx][0] > impact:
                        interest_temps[idx][0] = impact
                if impact > 0:
                    if interest_temps[idx][1] < impact:
                        interest_temps[idx][1] = impact
        # Compute the MAE matrix
        mae = []
        for ip in interest_temps:
            ip_mae = [abs((s[0]+s[1])-(ip[0]+ip[1])) for s in sensor_temps]
            mae.append(ip_mae)
        return mae

    def get_temp_at(self, x, y):
        """Determine the measured temperature at point x,y"""
        temp = [0,0]
        for t in self.temp_sources:
            dist = sqrt((t.x-x)**2 + (t.y-y)**2)
            impact = 0
            if dist <= 1:
                impact = t.delta
            elif dist < d_max:
                # Delta t is much smaller than distance: (-5,5) vs (0,500),
                # so we use log to reduce the ratio.
                # Also, the +1 is for avoiding higher impact than the delta
                impact = t.delta/(1+log(10, dist))
            # Only the most intense sources have effect
            if impact < 0:
                if temp[0] > impact:
                    temp[0] = impact
            if impact > 0:
                if temp[1] < impact:
                    temp[1] = impact
        return temp[0]+temp[1]

    def get_meas_error(self, sensor, interest_point):
        """Return the measurement error between the sensor and interest point"""
        s_x, s_y = sensor
        ip_x, ip_y = interest_point
        return abs(self.get_temp_at(s_x, s_y) - self.get_temp_at(ip_x, ip_y))

# -*- coding: utf-8 -*-
"""
Timer class.

Copyright Epic Games, Inc. All Rights Reserved.
"""

import time


class Timer:
    """ This class allows the user to record the time to execute a routine. """

    def __init__(self, unit='s'):
        """ Initialize the Timer class.

        Parameters (optional):
            unit (str) -- Time unit [s | ms | us | ns].
        """
        self.__unit = unit
        self.__state = 0

    @property
    def unit(self):
        """ Getter function for unit. """
        return self.__unit

    @unit.setter
    def unit(self, unit):
        """ Setter function for unit. """
        self.__unit = 's'
        if unit == 'm' or unit == 'ms' or unit == 'us' or unit == 'ns':
            self.__unit = unit

    def start(self):
        """ Start the clock. """
        self.__start = time.time()
        self.__state = 1   
    
    def stop(self):
        """ Stop the clock. """
        if self.__state == 1:
            self.__stop = time.time()
            
    def time_passed(self):
        """Get the current time passed since start."""
        if self.__state == 1:
            diff = time.time() - self.__start
            return self.convert_diff(diff)
        else:
            return 0.0
            
    def convert_diff(self, diff):
        """Get the difference in seconds, converted in the units of this timer.
        
            Parameters:
                 diff (seconds) -- The number of seconds passed.
                                 
            Returns:
                Time difference in user configured units.
        """
        if self.__unit == 'm':
            return diff / 60.0
        if self.__unit == 'ms':
            return diff * 1e3
        elif self.__unit == 'us':
            return diff * 1e6
        elif self.__unit == 'ns':
            return diff * 1e9
        else:
            return diff

    def show(self):
        """ Show the elapsed time. """
        diff = self.__stop - self.__start
        return self.convert_diff(diff)
      
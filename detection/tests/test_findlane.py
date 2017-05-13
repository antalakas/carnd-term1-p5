# -*- coding: utf-8 -*-


#///////////////////////////////////////////////////
#---------------------------------------------------
# File: test_findlane.py
# Author: Andreas Ntalakas
#---------------------------------------------------

#///////////////////////////////////////////////////
# Python
#---------------------------------------------------
import time
import string
import random

#///////////////////////////////////////////////////
# detection
#---------------------------------------------------
import numpy as np
from detection.detection import *

#---------------------------------------------------
import unittest

#///////////////////////////////////////////////////
class TestDetection(unittest.TestCase):
    """
    TestDetection
    """

    #///////////////////////////////////////////////////
    def setUp(self):
        self.f = Detection()

    def find_chessboard_corners(self):
        self.assertTrue(1 == 1)

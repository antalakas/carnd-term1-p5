# -*- coding: utf-8 -*-


#///////////////////////////////////////////////////
#---------------------------------------------------
# File: suite.py
# Author: Andreas Ntalakas
#---------------------------------------------------

#///////////////////////////////////////////////////
# Python
#---------------------------------------------------
import os
#---------------------------------------------------
import unittest

#///////////////////////////////////////////////////
# detection Tests
#---------------------------------------------------
from .test_detection import Detection
#---------------------------------------------------

#///////////////////////////////////////////////////
def test_suite():
    """
        `test_suite()` method creates a test suite
        for the unit-tests of detection package.
    """

    allTests = unittest.TestSuite()

    allTests.addTest(Detection('find_chessboard_corners'))

    return allTests

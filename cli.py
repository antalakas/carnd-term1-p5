#!/usr/bin/env python

import argparse
import findlane
from findlane.findlane.calibratecamera import CalibrateCamera
import os


#///////////////////////////////////////////////////
# Checks if the required modules have been installed.
def dependencies():
    try:
        return True
    except ImportError:
        return False


#///////////////////////////////////////////////////
#  FindLane command line interface
def findlane_cli():
    # ///////////////////////////////////////////////////
    parser = argparse.ArgumentParser(
        description="findlane_cli - A lightweight command-line wrapper for advanced lane finding.")

    parser.add_argument('--cal', action='store', dest='cal', help='Performs camera calibration')
    parser.add_argument('--imgf', action='store', dest='img', help='Executes findlane pipeline for test images')
    parser.add_argument('--vidf', action='store', dest='vid', help='Executes findlane pipeline for project video')

    # ///////////////////////////////////////////////////
    args = parser.parse_args()

    base_directory = os.path.dirname(os.path.abspath(__file__))

    if args.cal:
        cc = CalibrateCamera(base_directory)
        successfully_calibrated = cc.find_chessboard_corners(6, 9, False)
        print("successfully calibrated: %s images" % str(successfully_calibrated))
        cc.check_undistort(False)

    if args.img:
        fl = FindLane(base_directory)
        fl.execute_image_pipeline(True)

    if args.vid:
        fl = FindLane(base_directory)
        fl.project_video()

if __name__ == "__main__":

    try:
        if dependencies():
            findlane_cli()
        else:
            raise(Exception, "Packages required: ...")
    except KeyboardInterrupt:
        print("\n")

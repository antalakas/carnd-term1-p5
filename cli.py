#!/usr/bin/env python

import argparse
from findlane.findlane.calibratecamera import CalibrateCamera
from findlane.findlane.findlane import FindLane
from detection.detection.detection import Detection
import os


#///////////////////////////////////////////////////
# Checks if the required modules have been installed.
def dependencies():
    try:
        return True
    except ImportError:
        return False


#///////////////////////////////////////////////////
#  Vehicle Detection command line interface
def detection_cli():
    # ///////////////////////////////////////////////////
    parser = argparse.ArgumentParser(
        description="detection_cli - A lightweight command-line wrapper for Vehicle Detection.")

    parser.add_argument('--cal', action='store', dest='cal', help='Performs camera calibration')
    parser.add_argument('--imgf', action='store', dest='imgf', help='Executes findlane pipeline for test images')
    parser.add_argument('--vidf', action='store', dest='vidf', help='Executes findlane pipeline for project video')
    parser.add_argument('--imgd', action='store', dest='imgd', help='Executes detection pipeline for test images')
    parser.add_argument('--vidd', action='store', dest='vidd', help='Executes detection pipeline for project video')

    # ///////////////////////////////////////////////////
    args = parser.parse_args()

    base_directory = os.path.dirname(os.path.abspath(__file__))

    if args.cal:
        cc = CalibrateCamera(base_directory)
        successfully_calibrated = cc.find_chessboard_corners(6, 9, False)
        print("successfully calibrated: %s images" % str(successfully_calibrated))
        cc.check_undistort(False)

    if args.imgf:
        fl = FindLane(base_directory)
        fl.execute_image_pipeline(True)

    if args.vidf:
        fl = FindLane(base_directory)
        fl.project_video()

    if args.imgd:
        det = Detection(base_directory)
        det.execute_image_pipeline(True)

    if args.vidd:
        det = Detection(base_directory)
        det.project_video()

if __name__ == "__main__":

    try:
        if dependencies():
            detection_cli()
        else:
            raise(Exception, "Packages required: ...")
    except KeyboardInterrupt:
        print("\n")

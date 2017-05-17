import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import pickle
from moviepy.editor import VideoFileClip
# import time
from scipy.ndimage.measurements import label

import os

from .classifier import Classifier
from .features import Features
from .painter import Painter


class Detection(object):
    def __init__(self, base_directory, small=True, findlane=None):
        self.camera_calibration_path = os.path.join(base_directory, 'camera_cal')
        self.examples_path = os.path.join(base_directory, 'examples')
        self.output_images = os.path.join(base_directory, 'output_images')
        self.test_images = os.path.join(base_directory, 'test_images')

        self.input_video_path = os.path.join(base_directory, 'input_video')
        self.output_video_path = os.path.join(base_directory, 'output_video')

        self.color_space = 'YUV'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
        self.orient = 11           # HOG orientations
        self.pix_per_cell = 16     # HOG pixels per cell
        self.cell_per_block = 2    # HOG cells per block
        self.hog_channel = 'ALL'  # Can be 0, 1, 2, or "ALL"

        # self.ystart = 400
        # self.ystop = 656
        # self.scale = 1.5

        self.scaler = [(400, 464, 1.0),
                       (416, 480, 1.0),
                       (400, 496, 1.5),
                       (432, 528, 1.5),
                       (400, 528, 2.0),
                       (432, 560, 2.0),
                       (400, 630, 3.5),
                       (464, 694, 3.5)]

        self.bbox_history = []

        self.findlane = findlane
        self.features = Features()
        self.classifier = Classifier(base_directory, small)
        self.painter = Painter()

        calibration_path = os.path.join(self.camera_calibration_path, 'wide_dist_pickle.p')
        handle = open(calibration_path, 'rb')
        self.dist_pickle = pickle.load(handle)

    def add_bbox_history(self, bbox_list):
        self.bbox_history.append(bbox_list)

        if len(self.bbox_history) > 10:
            self.bbox_history.pop(0)

    def classifier_pipeline(self):
        self.classifier.load_data()

        self.classifier.extract_features(
            color_space=self.color_space,
             orient=self.orient,
             pix_per_cell=self.pix_per_cell,
             cell_per_block=self.cell_per_block,
             hog_channel=self.hog_channel)

        self.classifier.train()

    def single_image_pipeline(self):
        image_filename = os.path.join(self.test_images, 'test1.jpg')
        image = mpimg.imread(image_filename)
        plt.imshow(image)

        rectangles = []
        for scaler_item in self.scaler:
            bboxes = self.find_cars(image, scaler_item[0], scaler_item[1], scaler_item[2])
            rectangles.append(bboxes)

        # flatten list of lists
        rectangles = [item for sublist in rectangles for item in sublist]
        image_rect1 = self.painter.draw_boxes(image, rectangles)
        plt.imshow(image_rect1)

        heatmap = np.zeros_like(image[:, :, 0])
        heatmap = self.painter.add_heat(heatmap, rectangles)

        plt.figure(figsize=(10, 10))
        plt.imshow(heatmap, cmap='hot')

        heatmap = self.painter.apply_threshold(heatmap, 2)

        plt.figure(figsize=(10, 10))
        plt.imshow(heatmap, cmap='hot')

        labels = label(heatmap)

        plt.figure(figsize=(10, 10))
        plt.imshow(labels[0], cmap='gray')
        print(labels[1], 'cars found')

        image_rect = self.painter.draw_labeled_bboxes(np.copy(image), labels)

        plt.imshow(image_rect)

        plt.pause(5.001)
        # input("Press Enter to continue...")

    def image_pipeline(self):
        test_images = glob.glob(os.path.join(self.test_images, '*.jpg'))

        fig, axs = plt.subplots(5, 3, figsize=(16, 14))
        fig.subplots_adjust(hspace=.004, wspace=.002)
        axs = axs.ravel()

        for i, im in enumerate(test_images):
            image = mpimg.imread(im)

            rectangles = []
            for scaler_item in self.scaler:
                bboxes = self.find_cars(image, scaler_item[0], scaler_item[1], scaler_item[2])
                # bboxes = self.find_cars(image, self.ystart, self.ystop, self.scale)
                rectangles.append(bboxes)

            # flatten list of lists
            rectangles = [item for sublist in rectangles for item in sublist]
            # image_rect = self.painter.draw_boxes(image, rectangles)

            heatmap = np.zeros_like(image[:, :, 0])
            heatmap = self.painter.add_heat(heatmap, rectangles)
            heatmap = self.painter.apply_threshold(heatmap, 2)
            labels = label(heatmap)
            image_rect = self.painter.draw_labeled_bboxes(image, labels)

            axs[i].imshow(image_rect)
            axs[i].axis('off')

        plt.pause(5.001)
        # input("Press Enter to continue...")

    def execute_video_pipeline(self, image):
        undist_image = cv2.undistort(image, self.dist_pickle["mtx"], self.dist_pickle["dist"], None, self.dist_pickle["mtx"])
        # bboxes = self.find_cars(undist_image, 400, 656, 1.5)
        # return self.painter.draw_boxes(image, bboxes)

        rectangles = []
        for scaler_item in self.scaler:
            bboxes = self.find_cars(image, scaler_item[0], scaler_item[1], scaler_item[2])
            # bboxes = self.find_cars(image, self.ystart, self.ystop, self.scale)
            rectangles.append(bboxes)

        # flatten list of lists
        rectangles = [item for sublist in rectangles for item in sublist]

        self.add_bbox_history(rectangles)

        # image_rect = self.painter.draw_boxes(image, rectangles)
        heatmap = np.zeros_like(image[:, :, 0])

        for bbox_list in self.bbox_history:
            heatmap = self.painter.add_heat(heatmap, bbox_list)

        heatmap = self.painter.apply_threshold(heatmap, len(self.bbox_history)//2)
        labels = label(heatmap)
        return self.painter.draw_labeled_bboxes(image, labels)

    def project_video(self):
        # Execute the pipeline for the video file
        in_project_video = os.path.join(self.input_video_path, 'project_video.mp4')

        # project_clip = VideoFileClip(in_project_video).subclip(20, 26)
        # project_clip = VideoFileClip(in_project_video).subclip(35, 42)
        # project_clip = VideoFileClip(in_project_video).subclip(35, 45)
        # project_clip = VideoFileClip(in_project_video).subclip(41, 45)
        # project_clip = VideoFileClip(in_project_video).subclip(20, 45)
        project_clip = VideoFileClip(in_project_video)

        out_project_clip = project_clip.fl_image(self.execute_video_pipeline)  # NOTE: this function expects color images!!

        out_project_video = os.path.join(self.output_video_path, 'out_project_video.mp4')
        out_project_clip.write_videofile(out_project_video, audio=False)

    def execute_video_pipeline_with_findlane(self, image):
        # runs detection pipeline
        undist_image = cv2.undistort(image, self.dist_pickle["mtx"], self.dist_pickle["dist"], None, self.dist_pickle["mtx"])

        rectangles = []
        for scaler_item in self.scaler:
            bboxes = self.find_cars(image, scaler_item[0], scaler_item[1], scaler_item[2])
            # bboxes = self.find_cars(image, self.ystart, self.ystop, self.scale)
            rectangles.append(bboxes)

        # flatten list of lists
        rectangles = [item for sublist in rectangles for item in sublist]

        self.add_bbox_history(rectangles)

        # image_rect = self.painter.draw_boxes(image, rectangles)
        heatmap = np.zeros_like(image[:, :, 0])

        for bbox_list in self.bbox_history:
            heatmap = self.painter.add_heat(heatmap, bbox_list)

        heatmap = self.painter.apply_threshold(heatmap, len(self.bbox_history)//2)
        labels = label(heatmap)

        # runs lane find pipeline
        self.findlane.image_shape = undist_image.shape

        thresholded_binary, warped, out_img, ploty, left_fitx, right_fitx = \
            self.findlane.pipeline(
                undist_image,
                sobel_kernel_size=self.findlane.sobel_kernel_size,
                sx_thresh=self.findlane.sx_thresh,
                sy_thresh=self.findlane.sy_thresh,
                s_thresh=self.findlane.s_thresh,
                mag_thresh=self.findlane.mag_thresh,
                dir_thresh=self.findlane.dir_thresh,
                test_image_pipeline=False,
                visualize=False)

        result, newwarp = self.findlane.project_back(image, thresholded_binary, warped, ploty, left_fitx, right_fitx)

        img = self.findlane.plot_dashboard(result, ploty, left_fitx, right_fitx, newwarp)

        # combine detection with found line
        return self.painter.draw_labeled_bboxes(img, labels)

    def project_video_with_findlane(self):
        in_project_video = os.path.join(self.input_video_path, 'project_video.mp4')
        project_clip = VideoFileClip(in_project_video)

        out_project_clip = project_clip.fl_image(self.execute_video_pipeline_with_findlane)

        out_project_video = os.path.join(self.output_video_path, 'out_project_video.mp4')
        out_project_clip.write_videofile(out_project_video, audio=False)

    # Define a single function that can extract features using hog sub-sampling and make predictions
    def find_cars(self, img, ystart, ystop, scale):

        img = img.astype(np.float32) / 255

        img_tosearch = img[ystart:ystop, :, :]

        ctrans_tosearch = self.features.convert_color(img_tosearch, conv='RGB2YUV')

        if scale != 1:
            imshape = ctrans_tosearch.shape
            ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))

        ch1 = ctrans_tosearch[:, :, 0]
        ch2 = ctrans_tosearch[:, :, 1]
        ch3 = ctrans_tosearch[:, :, 2]

        # Define blocks and steps as above
        # nxblocks = (ch1.shape[1] // self.pix_per_cell) - self.cell_per_block + 1
        # nyblocks = (ch1.shape[0] // self.pix_per_cell) - self.cell_per_block + 1
        nxblocks = (ch1.shape[1] // self.pix_per_cell) + 1
        nyblocks = (ch1.shape[0] // self.pix_per_cell) + 1

        nfeat_per_block = self.orient * self.cell_per_block ** 2

        # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
        window = 64
        # nblocks_per_window = (window // self.pix_per_cell) - self.cell_per_block + 1
        nblocks_per_window = (window // self.pix_per_cell) - 1

        cells_per_step = 2  # Instead of overlap, define how many cells to step
        nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
        nysteps = (nyblocks - nblocks_per_window) // cells_per_step

        # Compute individual channel HOG features for the entire image
        hog1 = self.features.get_hog_features(ch1, self.orient, self.pix_per_cell, self.cell_per_block, feature_vec=False)
        hog2 = self.features.get_hog_features(ch2, self.orient, self.pix_per_cell, self.cell_per_block, feature_vec=False)
        hog3 = self.features.get_hog_features(ch3, self.orient, self.pix_per_cell, self.cell_per_block, feature_vec=False)

        bboxes = []
        for xb in range(nxsteps):
            for yb in range(nysteps):
                ypos = yb * cells_per_step
                xpos = xb * cells_per_step
                # Extract HOG for this patch
                hog_feat1 = hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                hog_feat2 = hog2[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                hog_feat3 = hog3[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

                xleft = xpos * self.pix_per_cell
                ytop = ypos * self.pix_per_cell

                test_prediction = self.classifier.svc.predict(hog_features)

                if test_prediction == 1:
                    xbox_left = np.int(xleft * scale)
                    ytop_draw = np.int(ytop * scale)
                    win_draw = np.int(window * scale)

                    bboxes.append(
                        ((xbox_left, ytop_draw + ystart), (xbox_left + win_draw, ytop_draw + win_draw + ystart)))

        return bboxes

    def visualize_hog(self):
        car_img = mpimg.imread(self.classifier.cars[0])
        _, car_dst = self.features.get_hog_features(car_img[:,:,2], 9, 8, 8, vis=True, feature_vec=True)
        noncar_img = mpimg.imread(self.classifier.notcars[0])
        _, noncar_dst = self.features.get_hog_features(noncar_img[:,:,2], 9, 8, 8, vis=True, feature_vec=True)

        # Visualize
        f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(7, 7))
        f.subplots_adjust(hspace = .4, wspace=.2)
        ax1.imshow(car_img)
        ax1.set_title('Example Car Image', fontsize=16)
        ax2.imshow(car_dst, cmap='gray')
        ax2.set_title('Example HOG Visualization', fontsize=16)
        ax3.imshow(noncar_img)
        ax3.set_title('Example Non-Car Image', fontsize=16)
        ax4.imshow(noncar_dst, cmap='gray')
        ax4.set_title('Example HOG Visualization', fontsize=16)
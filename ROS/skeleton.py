# !/usr/bin/env python3
import rospy
import numpy as np
import cv2

from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Header


def find_tv_contour(img):
    # Convert the image to grayscale
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    # equalize the histogram of the Y channel
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    # convert the YUV image back to RGB format
    image = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Perform adaptive thresholding to obtain a binary image
    _, threshold = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find contours in the binary image
    contours, hierarchy = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Find the index of the outermost contour (TV screen contour)
    tv_contour_index = -1
    max_area = 0
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            tv_contour_index = i

    # If the outermost contour is found
    if tv_contour_index != -1:
        # Create a mask image for the outermost contour
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, contours, tv_contour_index, (255), thickness=cv2.FILLED)

        # Find contours again within the mask image
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter the contours based on area to find the TV screen contour
        tv_contour = None
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 3000:  # Adjust this value based on the size of the TV screen in the image
                tv_contour = contour
                break

        if tv_contour is not None:
            # Get the pixels lying on the contour
            contour_mask = np.zeros_like(gray)
            cv2.drawContours(contour_mask, [tv_contour], -1, (255), thickness=1)

            # Extract the pixels lying on the contour
            contour_pixels = cv2.bitwise_and(image, image, mask=contour_mask)

            # Convert the image to the HSV color space
            hsv = cv2.cvtColor(contour_pixels, cv2.COLOR_BGR2HSV)

            # Define the lower and upper bounds for red color detection
            lower_red1 = np.array([0, 50, 50])
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([170, 50, 50])
            upper_red2 = np.array([180, 255, 255])

            # Create a mask for red color detection
            mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
            mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
            mask_red = cv2.bitwise_or(mask_red1, mask_red2)

            # Calculate the number of red and blue pixels
            red_pixels = np.sum(mask_red > 0)
            blue_pixels = np.sum(hsv[:, :, 0] > 100)

            # Check if red or blue color takes up more than half of the contour pixels
            cv2.drawContours(image, [tv_contour], -1, (0, 255, 0), 2)
            if red_pixels > len(contour_pixels) // 2:
                print("Red")
                return '-1'
            elif blue_pixels > len(contour_pixels) // 2:
                print("Blue")
                return '1'
            else:
                print("Neither")
                return '0'

    return None


class DetermineColor:
    def __init__(self):
        self.image_sub = rospy.Subscriber('/camera/color/image_raw', Image, self.callback)
        self.color_pub = rospy.Publisher('/rotate_cmd', Header, queue_size=10)
        self.bridge = CvBridge()
        self.count = 0

    def callback(self, data):
        try:
            # Listen to the image topic
            image = self.bridge.imgmsg_to_cv2(data, 'bgr8')
            cv2.imshow('Image', image)
            cv2.waitKey(1)

            # Prepare rotate_cmd msg
            # DO NOT DELETE THE BELOW THREE LINES!
            msg = Header()
            msg = data.header
            msg.frame_id = find_tv_contour(image)

            # Determine background color
            # TODO: Determine the color and assign +1, 0, or -1 for frame_id
            # msg.frame_id = '+1'  # CCW (Blue background)
            # msg.frame_id = '0'   # STOP
            # msg.frame_id = '-1'  # CW (Red background)

            # Publish color_state
            self.color_pub.publish(msg)

        except CvBridgeError as e:
            print(e)

    def rospy_shutdown(self, signal, frame):
        rospy.signal_shutdown("shut down")
        sys.exit(0)


if __name__ == '__main__':
    rospy.init_node('CompressedImages1', anonymous=False)
    detector = DetermineColor()
    rospy.spin()


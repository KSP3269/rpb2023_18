import rospy
import numpy as np
import cv2

from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Header

# Global variable to store the TV screen contour
tv_contour = None


def find_tv_contour(img):
    global tv_contour

    if tv_contour is not None:
        return tv_contour

    # Convert the image to grayscale
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    # equalize the histogram of the Y channel
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
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
        # Approximate the contour with simpler geometry (polygon)
        epsilon = 0.01 * cv2.arcLength(contours[tv_contour_index], True)
        approx = cv2.approxPolyDP(contours[tv_contour_index], epsilon, True)

        # Filter the approximated contour to have 4 sides
        if len(approx) == 4:
            tv_contour = approx

    return tv_contour


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

            # Retrieve the TV screen contour
            tv_contour = find_tv_contour(image)

            if tv_contour is not None:
                # Create a copy of the image to draw the contour on
                image_with_contour = image.copy()

                # Draw the contour on the image
                cv2.drawContours(image_with_contour, [tv_contour], -1, (0, 255, 0), 2)

                # Show the image with the contour
                cv2.imshow('Image with Contour', image_with_contour)
                cv2.waitKey(1)

                # Calculate the area of the TV screen
                # Calculate the area of the TV screen
                tv_screen_area = cv2.contourArea(tv_contour)

                # Convert the image to the HSV color space
                hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

                # Define the lower and upper bounds for red color detection
                lower_red1 = np.array([0, 50, 50])
                upper_red1 = np.array([10, 255, 255])
                lower_red2 = np.array([170, 50, 50])
                upper_red2 = np.array([180, 255, 255])

                # Define the lower and upper bounds for blue color detection
                lower_blue = np.array([90, 50, 50])
                upper_blue = np.array([130, 255, 255])

                # Create a mask for red and blue color detection
                mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
                mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
                mask_red = cv2.bitwise_or(mask_red1, mask_red2)

                mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

                # Calculate the number of red and blue pixels in the TV screen area
                red_pixels = np.sum(mask_red > 0)
                blue_pixels = np.sum(mask_blue > 0)

                # Check if red or blue color takes up more than half of the TV screen area
                if red_pixels > tv_screen_area / 2:
                    frame_id = '-1'  # Red background
                elif blue_pixels > tv_screen_area / 2:
                    frame_id = '+1'  # Blue background
                else:
                    frame_id = '0'  # Neither red nor blue background

                # Prepare rotate_cmd msg
                msg = Header()
                msg = data.header
                msg.frame_id = frame_id

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

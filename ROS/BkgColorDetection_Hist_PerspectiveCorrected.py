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
        # Create a mask image for the outermost contour
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, contours, tv_contour_index, (255), thickness=cv2.FILLED)

        # Find contours again within the mask image
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter the contours based on area to find the TV screen contour
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 3000:  # Adjust this value based on the size of the TV screen in the image
                tv_contour = contour
                break

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
            cv2.waitKey(1)

            # Retrieve the TV screen contour
            tv_contour = find_tv_contour(image)

            if tv_contour is not None:
                # Calculate the aspect ratio based on the provided screen ratio
                screen_ratio = 16 / 9
                aspect = screen_ratio / (cv2.arcLength(tv_contour, True) / cv2.contourArea(tv_contour)) 

                # Calculate the width of the transformed image
                tv_width = int(min(cv2.contourArea(tv_contour), cv2.arcLength(tv_contour, True) / aspect))

                # Calculate the height of the transformed image
                tv_height = int(tv_width / aspect)

                # Calculate the target points for perspective transformation
                target_points = np.array([[0, 0], [tv_width - 1, 0], [tv_width - 1, tv_height - 1], [0, tv_height - 1]], dtype=np.float32)

                # Reshape the image to match the aspect ratio of the TV screen
                tv_contour_reshaped = tv_contour.reshape(4, 2).astype(np.float32)
                transformed_image = cv2.warpPerspective(image, cv2.getPerspectiveTransform(tv_contour_reshaped, target_points), (tv_width, tv_height))


                # Rotate the transformed image clockwise by 90 degrees to make it upright
                transformed_image = cv2.rotate(transformed_image, cv2.ROTATE_90_CLOCKWISE)

                # Show the reshaped image
                cv2.imshow('Reshaped Image', transformed_image)
                cv2.waitKey(1)

                # Convert the reshaped image to the HSV color space
                hsv = cv2.cvtColor(transformed_image, cv2.COLOR_BGR2HSV)

                # Define the lower and upper bounds for red color detection
                lower_red1 = np.array([0, 50, 50])
                upper_red1 = np.array([13, 255, 255])
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

                # Highlight the red, blue, and neither pixels in the transformed image
                transformed_image_with_highlight = transformed_image.copy()
                transformed_image_with_highlight[mask_red > 0] = [0, 0, 255]  # Red pixels
                transformed_image_with_highlight[mask_blue > 0] = [255, 0, 0]  # Blue pixels
                transformed_image_with_highlight[np.logical_and(mask_red == 0, mask_blue == 0)] = [0, 255, 0] # Neither pixels

                # Show the transformed image with pixel highlights
                cv2.imshow('Transformed Image with Highlights', transformed_image_with_highlight)
                cv2.waitKey(1)

                # Determine the dominant color based on the number of red and blue pixels
                if red_pixels > blue_pixels:
                    color_state = '-1'  # Red background
                elif blue_pixels > red_pixels:
                    color_state = '+1'  # Blue background
                else:
                    color_state = '0'  # Neither red nor blue background

                # Publish the color state
                self.color_pub.publish(color_state)

        except CvBridgeError as e:
            print(e)


            # Check if red or blue color takes up more than half of the TV screen area
            if red_pixels > (tv_width * tv_height) / 2:
                frame_id = '-1'  # Red background
            elif blue_pixels > (tv_width * tv_height) / 2:
                frame_id = '+1'  # Blue background
            else:
                frame_id = '0'  # Neither red nor blue background

            # Prepare rotate_cmd msg
            msg = Header()
            msg.frame_id = frame_id

            # Publish the rotate_cmd msg
            self.color_pub.publish(msg)
        except CvBridgeError as e:
            print(e)
if __name__ == '__main__':
    rospy.init_node('CompressedImages1', anonymous=False)
    detector = DetermineColor()
    rospy.spin()

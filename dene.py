import numpy as np
import cv2

# Load an color image in grayscale
img = cv2.imread('um_000002.png')

def region_of_interest(img, vertices):
    #defining a blank mask to start with
    mask = np.zeros_like(img)

    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    #filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img


# Grayscale and hsv color space versions of the image
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

#choose only regions that are yellow and white since lines are 
# made up of these colors
lower_yellow = np.array([20, 100, 100], dtype = "uint8")
upper_yellow = np.array([30, 255, 255], dtype= "uint8")
mask_yellow = cv2.inRange(img_hsv, lower_yellow, upper_yellow)
mask_white = cv2.inRange(gray_image, 200, 255)
mask_yw = cv2.bitwise_or(mask_white, mask_yellow)
mask_yw_image = cv2.bitwise_and(gray_image, mask_yw)

# Gaussian blurring to the masked image in order to reduce
# noise in the output of Canny edge detector
kernel_size = 5
gauss_gray = cv2.GaussianBlur(mask_yw_image,(kernel_size,kernel_size),0)

# Edges found using Canny edge detector
low_threshold = 50
high_threshold = 150
canny_edges = cv2.Canny(gauss_gray,low_threshold,high_threshold)

imshape = img.shape
lower_left = [imshape[1]/9,imshape[0]]
lower_right = [imshape[1]-imshape[1]/9,imshape[0]]
top_left = [imshape[1]/2-imshape[1]/8,imshape[0]*5/10]
top_right = [imshape[1]/2+imshape[1]/8,imshape[0]*5/10]
vertices = [np.array([lower_left,top_left,top_right,lower_right],dtype=np.int32)]
roi_image = region_of_interest(canny_edges, vertices)

#rho and theta are the distance and angular resolution of the grid in Hough space
#same values as quiz
rho = 1
theta = np.pi/180
#threshold is minimum number of intersections in a grid for candidate line to go to output
threshold = 40
min_line_len = 50
max_line_gap = 200

line_image = hough_lines(roi_image, rho, theta, threshold, min_line_len, max_line_gap)
result = cv2.addWeighted(img, 1, line_image, 2, 0)

cv2.imshow('image',result)
cv2.waitKey(0)
cv2.destroyAllWindows()



# YERINI BIL






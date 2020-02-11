# VERSION: 1.00
# DATE: 2020-JAN-12
# Built image library v1.0 using this

from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2

# camera resolution
ROWS = 224
COLS = 224

# initialize the camera
camera = PiCamera()
camera.resolution = (ROWS, COLS)
camera.framerate = 30
camera.hflip = True
camera.vflip = True  # based on physical camera position
rawCapture = PiRGBArray(camera, size=(ROWS, COLS))    # originally 640,480
bgsub = cv2.createBackgroundSubtractorMOG2(history=30, varThreshold=30)

# camera warmup
time.sleep(0.1)

# filename for saving
str1 = 'else_'
num = 0
ext = '.jpg'

# capture frames from the camera
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    # grab the raw NumPy array representing the image, then initialize the timestamp
    # and occupied/unoccupied text
    image = frame.array
    image_bgsub = bgsub.apply(image)
    #################
    # later:
    # COPY 1 channel array to 3 channels
    # maybe np.repeat
    ############################

    fname = str1+str(num)+ext
    cv2.imwrite(fname, image_bgsub)  # writing as 'bgr' but it's all thresholded. should be fine?
    num += 1

    # show the frames
    cv2.imshow("Original", image)
    cv2.imshow("BG Subtracted", image_bgsub)
    key = cv2.waitKey(1) & 0xFF

    # delete frame so we can load next one
    rawCapture.truncate(0)

    # break conditions
    if key == ord("q") or num > 2000:
        break

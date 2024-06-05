import cv2
import numpy as np

# Define the dimensions of the board
board_width = 640
board_height = 480

# Define the dimensions of the markers
marker_size = 0.05

# Create a video capture object for the camera
vid = cv2.VideoCapture(0)

# Create a background subtractor object
back_sub = cv2.createBackgroundSubtractorMOG2()

# Create a marker detector object
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
parameters = cv2.aruco.DetectorParameters()

while True:
    # Capture the video frame by frame
    ret, frame = vid.read()

    if not ret:
        break

    # Convert the frame to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Apply background subtraction
    fg_mask = back_sub.apply(frame)

    # Apply thresholding to the HSV frame
    _, thresh = cv2.threshold(fg_mask, 25, 255, cv2.THRESH_BINARY)

    # Detect markers
    corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=parameters)

    # Draw the detected markers
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        # Estimate pose of each marker
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, marker_size, None, None)

        for rvec, tvec in zip(rvecs, tvecs):
            # Draw axis for each marker
            cv2.aruco.drawAxis(frame, None, None, rvec, tvec, 0.1)

    # Display the resulting frame
    cv2.imshow('frame', frame)

    # Check for the 'q' key to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()

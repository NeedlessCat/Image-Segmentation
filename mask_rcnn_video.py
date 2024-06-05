import cv2
import numpy as np

# Load the RCNN
net = cv2.dnn.readNetFromTensorflow("dnn/frozen_inference_graph.pb", "dnn/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt")

# Initialize Yt-1 as None before processing the first frame
Yt_minus_1 = None

# Define the UpdateFrame function
def UpdateFrame(Xt, Rt, Yt_minus_1):
    if Yt_minus_1 is None:
        Yt_minus_1 = Xt
    return (Xt & (~Rt))+(Yt_minus_1 & Rt)

# For Videos
cap = cv2.VideoCapture("video.mp4")
ret, img = cap.read()
height, width, _ = img.shape
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 'XVID' for .avi format
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (width, height))

while True:
    # Capture frame-by-frame
    ret, img = cap.read()
    
    # If no frame is captured or video ends, exit the loop
    if not ret:
        break  

    height, width, _ = img.shape

    # Create black image for the mask
    black_image = np.zeros((height, width, 3), np.uint8)

    # Detect Objects
    blob = cv2.dnn.blobFromImage(img, swapRB=True)
    net.setInput(blob)

    boxes, masks = net.forward(["detection_out_final", "detection_masks"])
    detection_count = boxes.shape[2]

    for i in range(detection_count):
        box = boxes[0, 0, i]
        class_id = box[1]
        score = box[2]
        if score < 0.5:
            continue

        # Get box coordinates
        x = int(box[3] * width)
        y = int(box[4] * height)
        x2 = int(box[5] * width)
        y2 = int(box[6] * height)

        roi = black_image[y: y2, x: x2]
        roi_height, roi_width, _ = roi.shape

        # Get the masks
        mask = masks[i, int(class_id)]
        mask = cv2.resize(mask, (roi_width, roi_height))
        _, mask = cv2.threshold(mask, 0.5, 255, cv2.THRESH_BINARY)

        # Get mask coordinates
        contours, _ = cv2.findContours(np.array(mask, np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            cv2.fillPoly(roi, [cnt], (255, 255, 255))

# Apply the segmentation mask to the current frame
    Rt = black_image
    Xt = img

    # Update the frame using the UpdateFrame function
    Yt = UpdateFrame(Xt, Rt, Yt_minus_1)

    # Save the current frame's output to be used as Yt-1 for the next frame
    Yt_minus_1 = Yt

    # # Display the resulting frame
    # cv2.imshow("Image", Yt)

    # # Press 'q' to exit the video stream
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break
    out.write(Yt)
    print("On Progress...")

# When everything is done, release the capture
cap.release()
# cv2.destroyAllWindows()
out.release()
print("Video processing complete. The output file has been saved.")

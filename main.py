import numpy as np
import cv2
import torch

cap = cv2.VideoCapture(0, cv2.CAP_ANY)
if not cap.isOpened():
    raise IOError('Cannot open webcam')
# Tracker settings
on_tracker = False
count = 0
model_step = 50
# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', device=0)
model.conf = .45
model.classes = [0]
image_size = 640
color = (255, 0, 0)
# model.max_det = 2

while True:
    # Get cam videoflow
    _, image = cap.read()
    # Flip image
    image = cv2.flip(image, 1)
    # Timer for calculate FPS
    timer = cv2.getTickCount()
    # If tracker is online
    if on_tracker:
        # print('Tracker')
        # Update tracker; check if it works and obtain new boxes
        on_tracker, bbox = multi_tracker.update(image)
        # For each tracking object
        for tracker_box in bbox:
            tracker_box = list(map(lambda x: int(x), tracker_box))
            # Calculate points for drawing box
            point_1 = (tracker_box[0], tracker_box[1])
            point_2 = (tracker_box[0] + tracker_box[2], tracker_box[1] + tracker_box[3])
            # Draw obtained boxes
            cv2.rectangle(image, point_1, point_2, color, 2, 1)
    # Else run model
    else:
        # Get model result
        results = model(image[..., ::-1], size=image_size)
        results = results.xyxy[0].cpu().detach().numpy()[:, :4].astype(np.uint32).tolist()
        # If boxes output is not empty
        if results:
            # print('Model')
            # Init multi-tracker
            multi_tracker = cv2.legacy.MultiTracker_create()
            # Add object boxes to tracker
            for model_box in results:
                # Init tracker for each object
                tracker = cv2.legacy.TrackerMOSSE_create()
                # Get box parameters
                x_min, y_min, x_max, y_max = model_box
                # Tracker accept box in format - x, y, width, height
                multi_tracker.add(tracker, image, (x_min, y_min, x_max - x_min, y_max - y_min))
                # Turn on tacking bool mark
                on_tracker = True
    # Each model_step-th step of loop we run model to renew tracking object
    count += 1
    if count % model_step == 0:
        on_tracker = False
    # Calculate FPS
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
    # Display it on frame
    cv2.putText(image, f'FPS: {fps:.0f}', (5, 25), cv2.FONT_HERSHEY_SIMPLEX, .75, (50, 170, 50), 2)
    cv2.imshow('Image', image)
    c = cv2.waitKey(1)
    if c == 27:
        break

cap.release()
cv2.destroyAllWindows()

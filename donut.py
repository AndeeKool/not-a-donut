import sys
import cv2
import cvlib as cv
from cvlib.object_detection import draw_bbox
import numpy as np

# Change the sample to 00 / 01 / 02 to see it work
image = cv2.imread("img/sample-02.jpg") 

# apply object detection
bbox, label, conf = cv.detect_common_objects(image)

print(bbox, label, conf)

index = 0
for b in bbox:
    startX = b[0]
    endX = b[1]

    startY = b[2]
    endY = b[3]

    cv2.rectangle(image, (startX, endX), (startY, endY), (50,0,200), 2)

    object_crop = np.copy(image[endX:endY, startX:startY])
    bx, label, confidence = cv.detect_common_objects(object_crop)

    print(confidence)
    print(label)

    idx = np.argmax(confidence)
    label = label[idx]

    if label == 'donut':
        label = 'this is a donut'
    else:
        label = 'this is NOT a donut'

    cv2.putText(image, label, (startX, endX), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 0, 200), 2)

cv2.imshow("donut detector", image)
cv2.waitKey()

# save output
cv2.imwrite("donut_detection.jpg", image)

# release resources
cv2.destroyAllWindows()
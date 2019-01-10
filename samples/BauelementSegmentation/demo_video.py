
import cv2
import numpy as np
from mrcnn.visualize_video import model, display_instances, class_names


VIDEO_NAME = 'kondensators.mp4'


capture = cv2.VideoCapture(VIDEO_NAME)
size = (
    int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
    int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
)
codec = cv2.VideoWriter_fourcc(*'DIVX')
output = cv2.VideoWriter('videofile_masked2.avi', codec, 10.0, size)

# i = 0
# frame_rate_divider = 3
while (capture.isOpened()):
    ret, frame = capture.read()
    if ret:

        # Add Mask to frames
        results = model.detect([frame], verbose=0)
        r = results[0]
        frame = display_instances(frame, r['rois'], r['masks'], r['class_ids'],
                                  class_names, r['scores'], title="Bauelement Detection")
        # if i % frame_rate_divider == 0:
            # frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
        output.write(frame)
        cv2.imshow('frame', frame)
        #     i += 1
        # else:
        #     i += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

capture.release()
output.release()
cv2.destroyAllWindows()

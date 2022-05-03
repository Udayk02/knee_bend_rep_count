import cv2
import mediapipe as mp
import numpy as np
import time


# calculating angle between the points - hip, knee and ankle
def calc_angle(a, b, c):
    radians = np.arctan2(c.y - b.y, c.x - b.x) - np.arctan2(a.y - b.y, a.x - b.x)
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360.0 - angle
        
    return angle

# pose model
mp_pose = mp.solutions.pose
# for drawing
mp_drawing = mp.solutions.drawing_utils

vcap = cv2.VideoCapture('KneeBendVideo.mp4')

# initializing model with detection confidence of 0.5 and tracking confidence of 0.5
# as static_image_mode isn't performing well enough on the given video, it is remain default to False
pose = mp_pose.Pose(min_detection_confidence = 0.5, min_tracking_confidence = 0.5)
  
# width and height of the frame
width  = vcap.get(3)    
height = vcap.get(4)
fps = vcap.get(cv2.CAP_PROP_FPS)

# timer start
start = 0
# timer end
end = 0
# count 
rep_count = 0
# initialize angle of the leg
pre = 180
# for feedback
f_start = 0
f_end = 0
flag = False

# output video in mp4 format
out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (int(width), int(height)))

while(vcap.isOpened()):
    ret, frame = vcap.read()
   
    if frame is not None:    

        # conversion of frame from BGR to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # processing the image using the model
        result = pose.process(image)

        # conversion of frame again to normal
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        try:
            landmarks = result.pose_landmarks.landmark

            # taking all the points required
            left_ankle = landmarks[23]
            left_knee = landmarks[25] 
            left_hip = landmarks[27]

            right_ankle = landmarks[24]
            right_knee = landmarks[26]
            right_hip = landmarks[28]

            # determining which leg is nearer to the camera based on the 'z' value
            # the leg with less 'z' value is nearer to the camera
            if left_knee.z < right_knee.z:

                # calculating the angle
                angle = calc_angle(left_ankle, left_knee, left_hip)

                # coordinated to display angle - knee coordinates
                coord = tuple(np.multiply([left_knee.x, left_knee.y], [width, height]).astype(int))

                cv2.putText(image, str(int(angle)), coord, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
            
            else:

                angle = calc_angle(right_ankle, right_knee, right_hip)

                coord = tuple(np.multiply([int(right_knee.x), int(right_knee.y)], [width, height]).astype(int))

                cv2.putText(image, str(int(angle)), coord, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
                
            # if there is a frame fluctuation, then the angle will change abruptly, so handling the timer there.
            # 25 is arbitrarily taken
            if pre - angle > 40:
                continue

            # if the angle is less than 140, timer starts
            if angle < 140:
                if start == 0:
                    start = time.time()
                
            # if the angle is greater than 140, timer ends
            # then we check if the time is less than 8 seconds - if yes - then rep count is not taken.
            # and feedback is given
            if angle >= 140:
                end = time.time()
                if(start != 0):
                    rep_time = end - start
                    if rep_time < 8:
                        f_start = time.time()        # feedback
                        cv2.putText(image, "Keep your knee bent", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 0, 255), 1, cv2.LINE_AA)
                        flag = True
                    else:
                        rep_count += 1
                    start = end = 0
        except Exception as e:
            start = end = 0
            print(e)

        # drawing landmarks
        mp_drawing.draw_landmarks(image, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # displaying the feedback for 3 seconds
        if(flag):
            cv2.putText(image, "Keep your knee bent", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
            f_end = time.time()
            if(f_end - f_start) > 3:
                flag = False
                f_end = f_start = 0

        # displaying rep count on the window
        cv2.putText(image, "Rep Count: " + str(rep_count), (400, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.imshow('Video', image)
        out.write(image)
    else:
        print("Frame is none")
    
    pre = angle
        
    if cv2.waitKey(22) & 0xFF == ord('q'):
        break

out.release()
vcap.release()
cv2.destroyAllWindows()
print("Video stop")

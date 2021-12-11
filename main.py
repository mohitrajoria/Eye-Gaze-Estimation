import cv2
import dlib
import numpy as np
import math


min_x = 1000000000000000000000
max_x = -10
min_y = 765324678655
max_y = -1.3833333333333333

threshold_value = 0.0

kernel = np.ones((9, 9), np.uint8)

left = [36, 37, 38, 39, 40, 41]
right = [42, 43, 44, 45, 46, 47]
list = []
list2 = []

file_cal = -1
file_work = -1

g = input(">> Enter (Y) for webcam or enter (N) for video file input: ")

if g=="N":
    file_cal = input(">> Enter calibration file name: ")
    file_work = input(">> Enter working file name: ")

def temp(x):
    pass

if g=="Y":
    file_cal=0
    file_work=0

cv2.namedWindow('Calibrate')
cv2.createTrackbar('threshold', 'Calibrate', 0, 255, temp)
cv2.createTrackbar('Switch','Calibrate',0,1,temp)

stream = cv2.VideoCapture(file_cal)


face_detect = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def npArray(shape, dtype="int"):
	coords = np.zeros((68, 2), dtype=dtype)
	for i in range(0, 68):
		coords[i] = (shape.part(i).x, shape.part(i).y)
	return coords

def get_eye_coordinates(thresh, mid, img, right=False,cir=True):
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    try:
        cnt = max(cnts, key = cv2.contourArea)
        M = cv2.moments(cnt)

        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])

        if right:
            cx += mid
        if cir:
            cv2.circle(img, (cx, cy), 4, (0, 0, 255), 2)
        return cx,cy
    except:
        return -1,-1

def build_mask(mask, side):
    points = [landmarks[i] for i in side]
    points = np.array(points, dtype=np.int32)
    mask = cv2.fillConvexPoly(mask, points, 255)
    return mask

def mid_point(x1,y1,x2,y2):
    return int((x1+x2)/2),int((y1+y2)/2)

def geometric_dist(x1,y1,x2,y2):
    return math.sqrt(math.pow(x1-x2, 2) + math.pow(y1-y2,2))

def blink_length(landmarks,a):
    midlu_x,midlu_y = mid_point(landmarks.part(37+a).x,landmarks.part(37+a).y,landmarks.part(38+a).x,landmarks.part(38+a).y)
    midld_x,midld_y = mid_point(landmarks.part(40+a).x,landmarks.part(40+a).y,landmarks.part(41+a).x,landmarks.part(41+a).y)
    # cv2.line(frame,(midld_x,midld_y),(midlu_x,midlu_y),(0,255,0),2)

    blink_length_v = geometric_dist(midld_x,midld_y,midlu_x,midlu_y)
    midlr_x,midlr_y = landmarks.part(39+a).x,landmarks.part(39+a).y
    midll_x,midll_y = landmarks.part(36+a).x,landmarks.part(36+a).y
    blink_length_h = geometric_dist(midlr_x,midlr_y,midll_x,midll_y)

    # cv2.putText(frame,str(blink_length_h/blink_length_v),(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)
    blink_length = blink_length_h/blink_length_v

    return blink_length

def get_ratio_horizontal(x1,x2,landmarks):
    dist_left_centre = geometric_dist(x1,0,landmarks[39][0],0)
    dist_left_away = geometric_dist(x1,0,landmarks[36][0],0)
    dist_right_away = geometric_dist(x2,0,landmarks[36+6][0],0)
    dist_right_centre = geometric_dist(x2, 0,landmarks[39+6][0], 0)
    dis_left_ratio = 0
    dis_right_ratio  = 0

    if dist_left_centre!=0:
        dis_left_ratio = dist_left_away/dist_left_centre
    if dist_right_centre!=0:
        dis_right_ratio =dist_right_away/dist_right_centre

    dis_ratio = (dis_left_ratio+dis_right_ratio)/2
    return dis_ratio

def get_ratio_verticle(y1,y2,landmarks):
    midlu_x,midlu_y = mid_point(landmarks.part(37).x,landmarks.part(37).y,landmarks.part(38).x,landmarks.part(38).y)
    midld_x,midld_y = mid_point(landmarks.part(40).x,landmarks.part(40).y,landmarks.part(41).x,landmarks.part(41).y)

    midru_x,midru_y = mid_point(landmarks.part(37+6).x,landmarks.part(37+6).y,landmarks.part(38+6).x,landmarks.part(38+6).y)
    midrd_x,midrd_y = mid_point(landmarks.part(40+6).x,landmarks.part(40+6).y,landmarks.part(41+6).x,landmarks.part(41+6).y)

    dist_left_up = geometric_dist(0,y1,0,midlu_y)
    dist_left_down = geometric_dist(0,y1,0,midld_y)
    dist_right_up = geometric_dist(0,y2,0,midru_y)
    dist_right_down = geometric_dist(0,y2,0,midrd_y)
    dis_left_ratio = 0
    dis_right_ratio  = 0

    if dist_left_up!=0:
        dis_left_ratio = dist_left_down/dist_left_up
    if dist_right_up!=0:
        dis_right_ratio =dist_right_down/dist_right_up
    dis_ratio = (dis_left_ratio+dis_right_ratio)/2
    return dis_ratio

while True:

    _, image = stream.read()
    if image is None:
        break

    image = cv2.flip(image,1)
    gray_img = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    faces = face_detect(gray_img)
    for face in faces:
        x1,y1 = face.left(), face.top()
        x2,y2 = face.right(), face.bottom()

        cv2.rectangle(image,(x1,y1),(x2,y2),(0,255,0),2)

        landmarks = predictor(gray_img,face)
        landmarks_init = landmarks
        landmarks = npArray(landmarks_init)

        mask_eye = np.zeros(image.shape[:2],dtype=np.uint8)
        mask_eye = build_mask(mask_eye,left)
        mask_eye = build_mask(mask_eye,right)
        mask_eye = cv2.dilate(mask_eye, kernel, 5)

        eyes = cv2.bitwise_and(image, image, mask=mask_eye)
        mask_eye = (eyes == [0, 0, 0]).all(axis=2)
        eyes[mask_eye] = [255, 255, 255]

        mid = (landmarks[42][0] + landmarks[39][0]) // 2
        eyes_gray = cv2.cvtColor(eyes, cv2.COLOR_BGR2GRAY)

        threshold_v = cv2.getTrackbarPos('threshold', 'Calibrate')
        _, threshold = cv2.threshold(eyes_gray, threshold_v, 255, cv2.THRESH_BINARY)
        threshold = cv2.erode(threshold, None, iterations=2)
        threshold = cv2.dilate(threshold, None, iterations=4)
        threshold = cv2.medianBlur(threshold, 3)
        threshold = cv2.bitwise_not(threshold)

        x1,y1 = get_eye_coordinates(threshold[:, 0:mid], mid, image)
        x2,y2 = get_eye_coordinates(threshold[:, mid:], mid,image, True)

        ss = cv2.getTrackbarPos('Switch','Calibrate')
        if ss==1:
            threshold_value = threshold_v
            if x1!=-1 and x2!=-1 and y1!=-1 and y2!=-1:
                dis_h = get_ratio_horizontal(x1,x2,landmarks)
                dis_v = get_ratio_verticle(y1,y2,landmarks_init)
                if dis_h!=0:
                    list.append(dis_h)
                if dis_v!=0:
                    list2.append(dis_v)

                # print(min(list),max(list),min(list2),max(list2))

                min_x = min(list)
                max_x = max(list)
                # angle = ((180)/(max_x-min_x))*(dis_h)
                min_y = min(list2)
                max_y = max(list2)
            # angle1 = ((180)/(max_y-min_y))*(dis_v)
            # cv2.putText(image,str(angle),(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)
            # cv2.putText(image,str(angle1),(50,250),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)
    cv2.imshow("Calibrate", image)

    key = cv2.waitKey(1)
    if key == 27:
        break

stream.release()
cv2.destroyAllWindows()
stream = cv2.VideoCapture(file_work)

while True:

    _, frame = stream.read()

    if frame is None:
        break

    frame = cv2.flip(frame,1)
    gray_img = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    faces = face_detect(gray_img)
    for face in faces:
        x11,y11 = face.left(), face.top()
        x21,y21 = face.right(), face.bottom()

        landmarks = predictor(gray_img,face)
        landmarks_init = landmarks
        landmarks = npArray(landmarks_init)

        mask_eye = np.zeros(frame.shape[:2],dtype=np.uint8)
        mask_eye = build_mask(mask_eye,left)
        mask_eye = build_mask(mask_eye,right)
        mask_eye = cv2.dilate(mask_eye, kernel, 5)

        eyes = cv2.bitwise_and(frame, frame, mask=mask_eye)
        mask_eye = (eyes == [0, 0, 0]).all(axis=2)
        eyes[mask_eye] = [255, 255, 255]
        mid = (landmarks[42][0] + landmarks[39][0]) // 2
        eyes_gray = cv2.cvtColor(eyes, cv2.COLOR_BGR2GRAY)
        # threshold_value = cv2.getTrackbarPos('threshold', 'image')

        _, threshold = cv2.threshold(eyes_gray, threshold_value, 255, cv2.THRESH_BINARY)
        threshold = cv2.erode(threshold, None, iterations=2) #1
        threshold = cv2.dilate(threshold, None, iterations=4) #2
        threshold = cv2.medianBlur(threshold, 3) #3
        threshold = cv2.bitwise_not(threshold)

        x1,y1 = get_eye_coordinates(threshold[:, 0:mid], mid, frame,False,False)
        x2,y2 = get_eye_coordinates(threshold[:, mid:], mid,frame, True,False)

        if x1!=-1 and x2!=-1 and y1!=-1 and y2!=-1:

            dis_h = get_ratio_horizontal(x1,x2,landmarks)
            dis_v = get_ratio_verticle(y1,y2,landmarks_init)

            angle=0
            # angle1=0
            if dis_h!=0:
                if dis_h <= 1:
                    angle = ((90)/(1-min_x))*(dis_h)
                else:
                    angle = 90+((90)/(max_x-1))*(dis_h)

            if angle >= 80 and angle<=110:
                cv2.putText(frame,'Center',(x11,y21+25),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)
            elif angle < 85:
                cv2.putText(frame,'Left',(x11,y21+25),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)
            else:
                cv2.putText(frame,'Right',(x11,y21+25),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)
            # if dis_v!=0:
                # if dis_v <= 1:
                    # angle1 = ((90)/(1-min_y))*(dis_v)
                # else:
                    # angle1 = 90+((90)/(max_y-1))*(dis_v)

            # print(dis_h,dis_v)
            # min_x = 0.875
            # max_x = 2.2142857142857144
            # angle = ((180)/(max_x-min_x))*(dis_h)
            # min_y = 0.75
            # max_y = 1.3833333333333333
            # angle1 = ((180)/(max_y-min_y))*(dis_v)

            cv2.putText(frame,str(angle)[0:6],(x11,y11-10),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)
            # cv2.putText(frame,str(angle1),(50,250),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)
        cv2.rectangle(frame,(x11,y11),(x21,y21),(0,255,0),2)

    cv2.imshow("Estimation", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

stream.release()
cv2.destroyAllWindows()

print("\nSuccess: Program Executed successfully")

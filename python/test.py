import pyopenface
import cv2
import os
import time

base_path = os.path.realpath(__file__)
base_path = base_path[:base_path.find('OpenFace')]

def main():
    of = pyopenface.Detector(base_path + "OpenFace/lib/local/LandmarkDetector/model/main_clnf_general.txt")
    img = cv2.imread(base_path + "OpenFace/samples/sample3.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    time_0 = time.time()
    max_face_rect = of.detect(img)
    #max_face_rect is a rectangle [left, top, right , bottom]
    time_1 = time.time()
    print "detect take {:.4f} Second".format(time_1-time_0)
    re = of.landmark(img, max_face_rect)
    time_2 = time.time()
    print "landmark take {:.4f} Second".format(time_2-time_1)
    keypoints = [(int(kp[0]), int(kp[1])) for kp in zip(re[0], re[1])]
    cv2.rectangle(img, (int(max_face_rect[0]),int(max_face_rect[1])), (int(max_face_rect[2]), int(max_face_rect[3])),(255, 0, 0), 2)
    for kp in keypoints:
       cv2.circle(img, kp, 2 , (255, 0, 0), 1)
    cv2.imwrite("result.jpg", img)

if __name__ == "__main__":
    main()

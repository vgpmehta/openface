import pyopenface
import cv2
import os
import time

base_path = os.path.realpath(__file__)
base_path = base_path[:base_path.find('OpenFace')]

debug = True

def main():
    detector = pyopenface.Detector(base_path + "OpenFace/lib/local/LandmarkDetector/model/main_clnf_general.txt")
    tracker = pyopenface.Tracker()
    img = cv2.imread(base_path + "OpenFace/samples/sample3.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_for_track = img.copy()
    time_0 = time.time()
    face_rects_max = detector.detect(img)
    #max_face_rect is a rectangle [left, top, right , bottom]
    time_1 = time.time()
    print "detect take {:.4f} Second".format(time_1-time_0)
    re = detector.landmark(img, face_rects_max)
    face_keypoints = [(int(kp[0]), int(kp[1])) for kp in zip(re[0], re[1])]
    time_2 = time.time()
    print "landmark take {:.4f} Second".format(time_2-time_1)
    for kp in face_keypoints:
        cv2.circle(img, kp, 2 , (255, 0, 0), 1)
    cv2.rectangle(img,
                  (int(face_rects_max[0]), int(face_rects_max[1])),
                  (int(face_rects_max[2]), int(face_rects_max[3])),
                  (255, 0, 0),
                  2)
    if debug:
        cv2.imwrite("detect_result.jpg", img)

    time_2 = time.time()
    tr_results = tracker.tracking(img_for_track)
    time_3 = time.time()
    print "tracking take {:.4f} Second".format(time_3-time_2)

    face_rects = tr_results[0]
    face_keypoints  = tr_results[1]
    face_keypoints = [(int(kp[0]), int(kp[1])) for kp in zip(face_keypoints[0], face_keypoints[1])]
    num_face = len(face_keypoints)/68
    for kp in face_keypoints:
        cv2.circle(img_for_track, kp, 2 , (255, 0, 0), 1)
    for i in range(num_face):
        cv2.rectangle(img_for_track,
                      (int(face_rects[i*4+0]), int(face_rects[i*4+1])),
                      (int(face_rects[i*4+2]), int(face_rects[i*4+3])),
                      (255, 0, 0),
                      2)

    if debug:
        cv2.imwrite("track_result.jpg", img_for_track)


if __name__ == "__main__":
    main()

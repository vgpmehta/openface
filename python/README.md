# Depend Boost-python and numpy
### On Ubuntu 14.04
```
    sudo apt-get install libboost-all-dev python-dev
```
### On Mac
```
   TODO
```

# PyOpenFace
## Detector
### class instance
    detector = pyopenface.Detector(base_path + "OpenFace/lib/local/LandmarkDetector/model/main_clnf_general.txt")
### get face rect
    max_face_rect = detector.detect(img_gray)
### get face landmark
    re = detector.landmark(img, max_face_rect)
    face_keypoints = [(int(kp[0]), int(kp[1])) for kp in zip(re[0], re[1])]

## Tracker()
### class instance
    tracker = pyopenface.Tracker()
### tracking with frame
    tr_results = tracker.tracking(img_gray)
### get face rect
    face_rects = tr_results[0]
### get face landmark
    face_keypoints = tr_results[1]
    face_keypoints = [(int(kp[0]), int(kp[1])) for kp in zip(face_keypoints[0], face_keypoints[1])]


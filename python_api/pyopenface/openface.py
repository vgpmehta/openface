import os
from collections import namedtuple

from _openface import (FaceModelParams, CLNF, DetectLandmarksInImage, DetectLandmarksInImageBounds,
    DetectLandmarksInVideo, DetectSingleFaceHAAR, DetectSingleFaceHOG, DetectSingleFaceMTCNN)
from .models import installed_mod_path, local_mod_path, get_mod_path

CLM_DETECTOR = 0
CLNF_DETECTOR = 1
CECLM_DETECTOR = 2

CLM_GENERAL = 'clm_general' # fastest, but least accurate
CLM_WILD = 'clm_wild' # same as CLM_GENERAL, but only trained on 300W dataset
CLNF_GENERAL = 'clnf_general' # medium speed/accuracy
CLNF_WILD = 'clnf_wild' # same as CLN_GENERAL, but only trained on 300W dataset
CLNF_MULTI_PIE = 'clnf_multi_pie' # same as CLN_GENERAL, but only trained on Multi-PIE
CECLM_GENERAL = 'ceclm_general' # Slowest but most accurate

landmark_detector_types = {
    CLM_GENERAL: CLM_DETECTOR,
    CLM_WILD: CLM_DETECTOR,
    CLNF_GENERAL: CLNF_DETECTOR,
    CLNF_WILD: CLNF_DETECTOR,
    CLNF_MULTI_PIE: CLNF_DETECTOR,
    CECLM_GENERAL: CECLM_DETECTOR
}

HAAR_DETECTOR = 0
HOG_SVM_DETECTOR = 1
MTCNN_DETECTOR = 2
face_detectors = [HAAR_DETECTOR, HOG_SVM_DETECTOR, MTCNN_DETECTOR]

# TODO: do these need to be made windows-compatible?
haar_detector_path = 'classifiers/haarcascade_frontalface_alt.xml'
mtcnn_detector_path = 'model/mtcnn_detector/MTCNN_detector.txt'


Rect = namedtuple("Rect", ['x', 'y', 'w', 'h'])


class FaceParams(object):

    def __init__(self):

        modpath = get_mod_path()

        init_mod = 'main_ceclm_general.txt' # model file required to init w/o error
        mod_loc = os.path.join(modpath, 'model', init_mod) 
        haar_path = os.path.join(modpath, haar_detector_path)
        mtcnn_path = os.path.join(modpath, mtcnn_detector_path)

        self._params = FaceModelParams([' ',
            '-mloc', mod_loc#,
            #'-fdloc', haar_path
        ])
        self._params.haar_path = os.path.join(modpath, haar_path)
        self._params.mtcnn_path = os.path.join(modpath, mtcnn_path)
        #self._params.curr_landmark_detector = CECLM_DETECTOR
        #self._params.curr_face_detector = MTCNN_DETECTOR


class FaceModel(object):

    def __init__(self, landmark_det=CECLM_GENERAL):

        try:
            self._landmark_det = landmark_det
            self._landmark_det_type = landmark_detector_types[landmark_det]
        except KeyError:
            raise ValueError("Unknown landmark detector model '{0}'".format(landmark_det))

        modpath = get_mod_path()
        modname = 'main_{0}.txt'.format(landmark_det)
        self._modloc = os.path.join(modpath, 'model', modname)
        self._mod = CLNF(self._modloc)

        # Preload MTCNN and HAAR face detectors
        haar_loc = os.path.join(modpath, haar_detector_path)
        mtcnn_loc = os.path.join(modpath, mtcnn_detector_path)
        self._mod.load_haar(haar_loc)
        self._mod.load_mtcnn(mtcnn_loc)


    @property
    def model_path(self):
        return self._modloc


def detect_face(image, model, face_det=MTCNN_DETECTOR):
    """Returns confidence in face detection & rectangle defining detected face,
    if a face is detected. Otherwise, returns (None, None).
    """
    if face_det == HAAR_DETECTOR:
        result = DetectSingleFaceHAAR(image, model._mod)
    elif face_det == HOG_SVM_DETECTOR:
        result = DetectSingleFaceHOG(image, model._mod)
    elif face_det == MTCNN_DETECTOR:
        result = DetectSingleFaceMTCNN(image, model._mod)
    else:
        raise RuntimeError("Unknown face detector {0}".format(str(face_det)))

    if len(result):
        if face_det == HAAR_DETECTOR:
            confidence = None
            bbox = result
        else:
            # NOTE: HOG confidence has no upper bound, can be > 1.0 
            confidence = result[0]
            bbox = result[1:]
        return (confidence, Rect(*bbox))
    else:
        return (None, None)
    

def detect_landmarks(image, model, params, bbox = None):
    # TODO: array typechecking
    if bbox and len(bbox) == 4:
        return DetectLandmarksInImageBounds(image, model._mod, params._params, bbox)
    return DetectLandmarksInImage(image, model._mod, params._params)


def detect_landmarks_video(frame, model, params):
    # TODO: array typechecking
    return DetectLandmarksInVideo(frame, model._mod, params._params) 

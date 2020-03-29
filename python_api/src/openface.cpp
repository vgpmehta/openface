#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <opencv2/opencv.hpp>

#include "LandmarkDetectorModel.h"
#include "LandmarkDetectorParameters.h"
#include "LandmarkDetectorFunc.h"
#include "LandmarkDetectorUtils.h"

namespace py = pybind11;


cv::Mat numpy_uint8_to_mat(py::array_t<unsigned char>& input, bool togray) {

    py::buffer_info buf = input.request();

    // TODO extend this a bunch
    int dtype = CV_8UC1;
    if (input.ndim() == 3)
        dtype = (buf.shape[2] == 3) ? CV_8UC3 : CV_8UC4;

    cv::Mat mat(buf.shape[0], buf.shape[1], dtype, (unsigned char*)buf.ptr);

    // if input not grayscale, convert to grayscale before returning
    if (togray && dtype != CV_8UC1) {
        cv::Mat gray;
        int convtype = (dtype == CV_8UC3) ? cv::COLOR_BGR2GRAY : cv::COLOR_BGRA2GRAY;
        cv::cvtColor(mat, gray, convtype);
        return gray;
    }
    
    return mat;
}


PYBIND11_MODULE(_openface, m) {

    py::class_<LandmarkDetector::FaceModelParameters>(m, "FaceModelParams")
        .def(py::init<>())
        .def(py::init<std::vector<std::string> &>())
        .def_readwrite("limit_pose", &LandmarkDetector::FaceModelParameters::limit_pose)
        .def_readwrite("validate_detections", &LandmarkDetector::FaceModelParameters::validate_detections)
        .def_readwrite("use_face_template", &LandmarkDetector::FaceModelParameters::use_face_template)
        .def_readwrite("model_path", &LandmarkDetector::FaceModelParameters::model_location)
        .def_readwrite("haar_path", &LandmarkDetector::FaceModelParameters::haar_face_detector_location)
        .def_readwrite("mtcnn_path", &LandmarkDetector::FaceModelParameters::mtcnn_face_detector_location)
        .def_readwrite("curr_landmark_detector", &LandmarkDetector::FaceModelParameters::curr_landmark_detector)
        .def_readwrite("curr_face_detector", &LandmarkDetector::FaceModelParameters::curr_face_detector)
    ;

    py::class_<LandmarkDetector::CLNF>(m, "CLNF")
        .def(py::init<const std::string &>())
        .def_readonly("detection_success", &LandmarkDetector::CLNF::detection_success)
        .def_readonly("tracking_initialised", &LandmarkDetector::CLNF::tracking_initialised)
        .def_readonly("detection_certainty", &LandmarkDetector::CLNF::detection_certainty)
        .def_readonly("eye_model", &LandmarkDetector::CLNF::eye_model)
        .def_readonly("consecutive_failures", &LandmarkDetector::CLNF::failures_in_a_row)
        .def_readonly("loaded_successfully", &LandmarkDetector::CLNF::loaded_successfully)
        //.def_readonly("detected_landmarks", &LandmarkDetector::CLNF::detected_landmarks)
        .def_readonly("model_likelihood", &LandmarkDetector::CLNF::model_likelihood)
        .def_readonly("landmark_likelihoods", &LandmarkDetector::CLNF::landmark_likelihoods)
        .def("load_mtcnn", [](LandmarkDetector::CLNF& o, std::string& path) {
            o.face_detector_MTCNN.Read(path);
            bool ret = !o.face_detector_MTCNN.empty();
            return ret;
        })
        .def("load_haar", [](LandmarkDetector::CLNF& o, std::string& path) {
            o.face_detector_HAAR.load(path);
            bool ret = !o.face_detector_HAAR.empty();
            return ret;
        })
        .def("reset", [](LandmarkDetector::CLNF& o) {
            o.Reset();
            return py::cast<py::none>(Py_None);
        })
    ;

    m.def("DetectSingleFaceHAAR",
        [](py::array_t<unsigned char>& image,
           LandmarkDetector::CLNF& model
          ) {
              // NOTE: min_width and roi not implemented yet
              cv::Rect_<float> bbox;
              cv::Mat img = numpy_uint8_to_mat(image, true);
              cv::Point prefpoint = cv::Point(-1, -1);

              model.Reset();
              bool success = LandmarkDetector::DetectSingleFace(
                  bbox, img, model.face_detector_HAAR, prefpoint
              );
              if (success) {
                std::vector<float> vec = {bbox.x, bbox.y, bbox.width, bbox.height};
                return vec;
              } else {
                std::vector<float> vec = {};
                return vec;
              }
          }
    );

    m.def("DetectSingleFaceHOG",
       [](py::array_t<unsigned char>& image,
          LandmarkDetector::CLNF& model
         ) {
             // NOTE: min_width and roi not implemented yet
             cv::Rect_<float> bbox;
             cv::Mat img = numpy_uint8_to_mat(image, true);
             float confidence = 0.0;
             cv::Point prefpoint = cv::Point(-1, -1);

             model.Reset();
             bool success = LandmarkDetector::DetectSingleFaceHOG(
                 bbox, img, model.face_detector_HOG, confidence, prefpoint
             );
             if (success) {
               std::vector<float> vec = {confidence, bbox.x, bbox.y, bbox.width, bbox.height};
               return vec;
             } else {
               std::vector<float> vec = {};
               return vec;
             }
         }
    );

    m.def("DetectSingleFaceMTCNN",
        [](py::array_t<unsigned char>& image,
           LandmarkDetector::CLNF& model
          ) {
              // NOTE: min_width and roi not implemented yet
              cv::Rect_<float> bbox;
              cv::Mat img = numpy_uint8_to_mat(image, true);
              float confidence = 0.0;
              cv::Point prefpoint = cv::Point(-1, -1);

              model.Reset();
              bool success = LandmarkDetector::DetectSingleFaceMTCNN(
                  bbox, img, model.face_detector_MTCNN, confidence, prefpoint
              );
              if (success) {
                std::vector<float> vec = {confidence, bbox.x, bbox.y, bbox.width, bbox.height};
                return vec;
              } else {
                std::vector<float> vec = {};
                return vec;
              }
          }
    );

    m.def("DetectLandmarksInImage",
        [](py::array_t<unsigned char>& image,
           LandmarkDetector::CLNF& model,
           LandmarkDetector::FaceModelParameters& params
          ) {
              model.Reset();
              cv::Mat img = numpy_uint8_to_mat(image, true);
              if (LandmarkDetector::DetectLandmarksInImage(img, model, params, img)) {
                std::vector<float> vec = model.detected_landmarks;
                return vec;
              } else {
                std::vector<float> vec = {};
                return vec;
              }
          }
    );

    m.def("DetectLandmarksInImageBounds",
        [](py::array_t<unsigned char>& image,
           LandmarkDetector::CLNF& model,
           LandmarkDetector::FaceModelParameters& params,
           std::vector<float>& bbox
          ) {
              model.Reset();
              cv::Mat img = numpy_uint8_to_mat(image, true);
              cv::Rect_<float> bounds = cv::Rect_<float>(bbox[0], bbox[1], bbox[2], bbox[3]);
              if (LandmarkDetector::DetectLandmarksInImage(img, bounds, model, params, img)) {
                std::vector<float> vec = model.detected_landmarks;
                return vec;
              } else {
                std::vector<float> vec = {};
                return vec;
              }
          }
    );

    m.def("DetectLandmarksInVideo",
        [](py::array_t<unsigned char>& image,
           LandmarkDetector::CLNF& model,
           LandmarkDetector::FaceModelParameters& params
          ) {
              cv::Mat img = numpy_uint8_to_mat(image, true);
              if (LandmarkDetector::DetectLandmarksInVideo(img, model, params, img)) {
                std::vector<float> vec = model.detected_landmarks;
                return vec;
              } else {
                std::vector<float> vec = {};
                return vec;
              }
          }
    );

}
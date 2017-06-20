#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <iostream>
#include <boost/python.hpp>
#include <numpy/arrayobject.h>
#include <opencv2/core.hpp>
#include "detector.hpp"

namespace bp = boost::python;

bp::object landmark(Detector *detector, bp::object frame_obj, const bp::list& rect_object)
{

  PyArrayObject* frame_arr = reinterpret_cast<PyArrayObject*>(frame_obj.ptr());
  const int height = PyArray_DIMS(frame_arr)[0];
  const int width = PyArray_DIMS(frame_arr)[1];
  cv::Mat frame(cv::Size(width, height), CV_8UC1, PyArray_DATA(frame_arr), cv::Mat::AUTO_STEP);

  //rect_object[left, top, right, bottom]
  bp::ssize_t rect_len = bp::len(rect_object);
  assert(rect_len==4);
  double rect_x = bp::extract<double>(rect_object[0]);
  double rect_y = bp::extract<double>(rect_object[1]);
  double rect_height = bp::extract<double>(rect_object[2]) - bp::extract<double>(rect_object[0]);
  double rect_width  = bp::extract<double>(rect_object[3]) - bp::extract<double>(rect_object[1]);
  cv::Rect_<double> face_rect(rect_x, rect_y, rect_height, rect_width);

  cv::Mat_<double> landmarks = detector->Run(frame, face_rect);

  long int landmarks_size[2] = {landmarks.rows, landmarks.cols};
  PyObject * landmarks_obj = PyArray_SimpleNewFromData( 2, landmarks_size, NPY_DOUBLE, landmarks.data);
  bp::handle<> landmarks_handle(landmarks_obj);
  bp::numeric::array landmarks_arr(landmarks_handle);

  return landmarks_arr.copy();

}

bp::list detect(Detector *detector, bp::object frame_obj)
{

    PyArrayObject *frame_arr = reinterpret_cast<PyArrayObject *>(frame_obj.ptr());
    const int height = PyArray_DIMS(frame_arr)[0];
    const int width = PyArray_DIMS(frame_arr)[1];
    cv::Mat grayscale_frame(cv::Size(width, height), CV_8UC1, PyArray_DATA(frame_arr), cv::Mat::AUTO_STEP);
    cv::Rect_<double> face_rect = detector->DetectFace(grayscale_frame);
    bp::list list;
    list.append(face_rect.x);
    list.append(face_rect.y);
    list.append(face_rect.x+face_rect.height);
    list.append(face_rect.y+face_rect.width);
    //return list is face_rect[left, top, right, bottom]
    return list;

}

BOOST_PYTHON_MODULE(pyopenface) {

  bp::numeric::array::set_module_and_type("numpy", "ndarray");
  import_array();
  
  bp::class_<Detector>("Detector", bp::no_init)
      .def("__init__", bp::make_constructor(&Detector::Create))
      .def("detect",  &detect)
      .def("landmark", &landmark)
      ;

}

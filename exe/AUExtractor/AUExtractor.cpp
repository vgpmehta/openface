///////////////////////////////////////////////////////////////////////////////
// Copyright (C) 2017, Carnegie Mellon University and University of Cambridge,
// all rights reserved.
//
// ACADEMIC OR NON-PROFIT ORGANIZATION NONCOMMERCIAL RESEARCH USE ONLY
//
// BY USING OR DOWNLOADING THE SOFTWARE, YOU ARE AGREEING TO THE TERMS OF THIS LICENSE AGREEMENT.
// IF YOU DO NOT AGREE WITH THESE TERMS, YOU MAY NOT USE OR DOWNLOAD THE SOFTWARE.
//
// License can be found in OpenFace-license.txt

//     * Any publications arising from the use of this software, including but
//       not limited to academic journal and conference publications, technical
//       reports and manuals, must cite at least one of the following works:
//
//       OpenFace 2.0: Facial Behavior Analysis Toolkit
//       Tadas Baltru�aitis, Amir Zadeh, Yao Chong Lim, and Louis-Philippe Morency
//       in IEEE International Conference on Automatic Face and Gesture Recognition, 2018
//
//       Convolutional experts constrained local model for facial landmark detection.
//       A. Zadeh, T. Baltru�aitis, and Louis-Philippe Morency,
//       in Computer Vision and Pattern Recognition Workshops, 2017.
//
//       Rendering of Eyes for Eye-Shape Registration and Gaze Estimation
//       Erroll Wood, Tadas Baltru�aitis, Xucong Zhang, Yusuke Sugano, Peter Robinson, and Andreas Bulling
//       in IEEE International. Conference on Computer Vision (ICCV),  2015
//
//       Cross-dataset learning and person-specific normalisation for automatic Action Unit detection
//       Tadas Baltru�aitis, Marwa Mahmoud, and Peter Robinson
//       in Facial Expression Recognition and Analysis Challenge,
//       IEEE International Conference on Automatic Face and Gesture Recognition, 2015
//
///////////////////////////////////////////////////////////////////////////////
// FaceLandmarkImg.cpp : Defines the entry point for the console application for detecting landmarks in images.

// dlib
#include <dlib/image_processing/frontal_face_detector.h>

#include "LandmarkCoreIncludes.h"

#include <FaceAnalyser.h>
#include <GazeEstimation.h>

#include <ImageCapture.h>
#include <Visualizer.h>
#include <VisualizationUtils.h>
#include <RecorderOpenFace.h>
#include <RecorderOpenFaceParameters.h>

// zmq
#include <string>
#include <chrono>
#include <thread>
#include <iostream>
#include <zmq.hpp>
#include <string.h>
#include <stdio.h>
#include <unistd.h>

#include "base64.h"

#include <opencv2/imgproc/imgproc.hpp>

#include <sys/time.h>

#ifndef CONFIG_DIR
#define CONFIG_DIR "~"
#endif

using namespace std;

std::vector<std::string> get_arguments(int argc, char **argv)
{
  std::vector<std::string> arguments;

  for (int i = 0; i < argc; ++i)
  {
    arguments.push_back(std::string(argv[i]));
  }

  return arguments;
}

string convertToJSON(cv::Rect_<float> roi, vector<pair<string, double>> intensity, vector<pair<string, double>> presence)
{
  std::string json = "{";

  json = json + "\"roi\":{\"x\":" + to_string((int) roi.x) + ",\"y\":" + to_string((int) roi.y) +
    ",\"width\":" + to_string((int) roi.width) + ",\"height\":" + to_string((int) roi.height) + "},";

  json = json + "\"intensity\":{";

  for (int i = 0; i < intensity.size(); i++)
  {
    json = json + "\"" + intensity[i].first + "\":" + to_string(intensity[i].second);

    if (i < intensity.size() - 1)
    {
      json = json + ",";
    }
  }

  json = json + "},\"presence\":{";

  for (int i = 0; i < presence.size(); i++)
  {
    json = json + "\"" + presence[i].first + "\":" + to_string(presence[i].second);

    if (i < presence.size() - 1)
    {
      json = json + ",";
    }
  }

  json = json + "}}";

  return json;
}

string convertToJSON(cv::Rect_<float> roi)
{
  std::string json = "{";
    
  json = json + "\"roi\":{\"x\":" + to_string((int) roi.x) + ",\"y\":" + to_string((int) roi.y) +
    ",\"width\":" + to_string((int) roi.width) + ",\"height\":" + to_string((int) roi.height) + "},";

  json = json + "\"intensity\":{\"AU01\":\"-\",\"AU02\":\"-\",\"AU04\":\"-\",\"AU05\":\"-\"," +
    "\"AU06\":\"-\",\"AU07\":\"-\",\"AU09\":\"-\",\"AU10\":\"-\",\"AU12\":\"-\",\"AU14\":\"-\"," +
    "\"AU15\":\"-\",\"AU17\":\"-\",\"AU20\":\"-\",\"AU23\":\"-\",\"AU25\":\"-\",\"AU26\":\"-\",\"AU45\":\"-\"},";

  json = json + "\"presence\":{\"AU01\":\"-\",\"AU02\":\"-\",\"AU04\":\"-\",\"AU05\":\"-\",\"AU06\":\"-\",\"AU07\":\"-\"," +
    "\"AU09\":\"-\",\"AU10\":\"-\",\"AU12\":\"-\",\"AU14\":\"-\",\"AU15\":\"-\",\"AU17\":\"-\",\"AU20\":\"-\"," +
    "\"AU23\":\"-\",\"AU25\":\"-\",\"AU26\":\"-\",\"AU28\":\"-\",\"AU45\":\"-\"}";
  
  json = json + "}";

  return json;
}

string convertToJSON()
{
  std::string json = "{";
    
  json = json + "\"intensity\":{\"AU01\":\"-\",\"AU02\":\"-\",\"AU04\":\"-\",\"AU05\":\"-\"," +
    "\"AU06\":\"-\",\"AU07\":\"-\",\"AU09\":\"-\",\"AU10\":\"-\",\"AU12\":\"-\",\"AU14\":\"-\"," +
    "\"AU15\":\"-\",\"AU17\":\"-\",\"AU20\":\"-\",\"AU23\":\"-\",\"AU25\":\"-\",\"AU26\":\"-\",\"AU45\":\"-\"},";

  json = json + "\"presence\":{\"AU01\":\"-\",\"AU02\":\"-\",\"AU04\":\"-\",\"AU05\":\"-\",\"AU06\":\"-\",\"AU07\":\"-\"," +
    "\"AU09\":\"-\",\"AU10\":\"-\",\"AU12\":\"-\",\"AU14\":\"-\",\"AU15\":\"-\",\"AU17\":\"-\",\"AU20\":\"-\"," +
    "\"AU23\":\"-\",\"AU25\":\"-\",\"AU26\":\"-\",\"AU28\":\"-\",\"AU45\":\"-\"}";
  
  json = json + "}";

  return json;
}

void loadFaceModel(LandmarkDetector::CLNF &clnf_model, LandmarkDetector::FaceModelParameters &params)
{
  if (clnf_model.face_detector_HAAR.empty() && params.curr_face_detector == params.HAAR_DETECTOR)
  {
    clnf_model.face_detector_HAAR.load(params.haar_face_detector_location);
    clnf_model.haar_face_detector_location = params.haar_face_detector_location;
  }

  if (clnf_model.face_detector_MTCNN.empty() && params.curr_face_detector == params.MTCNN_DETECTOR)
  {
    clnf_model.face_detector_MTCNN.Read(params.mtcnn_face_detector_location);
    clnf_model.mtcnn_face_detector_location = params.mtcnn_face_detector_location;

    // If the model is still empty default to HOG
    if (clnf_model.face_detector_MTCNN.empty())
    {
      std::cout << "INFO: defaulting to HOG-SVM face detector" << std::endl;
      params.curr_face_detector = LandmarkDetector::FaceModelParameters::HOG_SVM_DETECTOR;
    }
  }

  // Warm up the models
  std::cout << "Warming up the models..." << std::endl;
  cv::Mat dummy_image = cv::Mat::zeros(cv::Size(640, 480), CV_8UC3);
  for (int i = 0; i < 10; ++i)
  {
    LandmarkDetector::DetectLandmarksInVideo(dummy_image, clnf_model, params, dummy_image);
  }
  std::cout << "Warm up runs are completed" << std::endl;

  // Reset the model parameters
  clnf_model.Reset();
}

cv::Rect_<float> detectSingleFace(const cv::Mat &rgb_image, LandmarkDetector::CLNF &clnf_model, LandmarkDetector::FaceModelParameters &params, cv::Mat &grayscale_image)
{
  cv::Rect_<float> bounding_box(0, 0, 0, 0);

  if (params.curr_face_detector == LandmarkDetector::FaceModelParameters::HOG_SVM_DETECTOR)
  {
    float confidence;
    LandmarkDetector::DetectSingleFaceHOG(bounding_box, grayscale_image, clnf_model.face_detector_HOG, confidence);
  }
  else if (params.curr_face_detector == LandmarkDetector::FaceModelParameters::HAAR_DETECTOR)
  {
    LandmarkDetector::DetectSingleFace(bounding_box, rgb_image, clnf_model.face_detector_HAAR);
  }
  else if (params.curr_face_detector == LandmarkDetector::FaceModelParameters::MTCNN_DETECTOR)
  {
    float confidence;
    LandmarkDetector::DetectSingleFaceMTCNN(bounding_box, rgb_image, clnf_model.face_detector_MTCNN, confidence);
  }

  // Add some buffer for the bounding_box
  int buffer = max(bounding_box.width, bounding_box.height) / 4;
  bounding_box.x = bounding_box.x - buffer;
  bounding_box.y = bounding_box.y - buffer;
  bounding_box.width = bounding_box.width + (2 * buffer);
  bounding_box.height = bounding_box.height + (2 * buffer);

  // Cast bounding_box to integer
  bounding_box.x = (int) bounding_box.x - 1;
  bounding_box.y = (int) bounding_box.y - 1;
  bounding_box.width = (int) bounding_box.width + 2;
  bounding_box.height = (int) bounding_box.height + 2;

  // Check image boundaries
  bounding_box.x = bounding_box.x > 0 ? bounding_box.x : 0;
  bounding_box.y = bounding_box.y > 0 ? bounding_box.y : 0;
  bounding_box.width = (bounding_box.x + bounding_box.width) < rgb_image.size().width ? bounding_box.width : (rgb_image.size().width - bounding_box.x);
  bounding_box.height = (bounding_box.y + bounding_box.height) < rgb_image.size().height ? bounding_box.height : (rgb_image.size().height - bounding_box.y);

  return bounding_box;
}

int main(int argc, char **argv)
{
  // Convert arguments to more convenient vector form
  std::vector<std::string> arguments = get_arguments(argc, argv);

  // Load the models
  LandmarkDetector::FaceModelParameters det_parameters(arguments);

  // The modules that are being used for tracking
  std::cout << "Loading landmark model" << std::endl;
  LandmarkDetector::CLNF face_model(det_parameters.model_location);

  loadFaceModel(face_model, det_parameters);

  if (!face_model.loaded_successfully)
  {
    std::cout << "ERROR: Could not load the landmark detector" << std::endl;
    return 1;
  }

  std::cout << "Loading facial feature extractor" << std::endl;
  FaceAnalysis::FaceAnalyserParameters face_analysis_params(arguments);
  face_analysis_params.OptimizeForImages();
  FaceAnalysis::FaceAnalyser face_analyser(face_analysis_params);

  std::cout << "Everything loaded" << std::endl;

  // ZMQ preparation
  zmq::context_t ctx;
  zmq::socket_t sock(ctx, ZMQ_PAIR);
  std::string addr{"localhost"};
  std::string port{argc > 1 ? arguments[1] : "5555"};
  std::cout << "Connecting socket on " + addr + ":" + port << std::endl;
  sock.connect("tcp://" + addr + ":" + port);

  cv::Rect_<float> roi(0, 0, 0, 0);
  cv::Mat greyScale_image, rgb_image_roi, greyScale_image_roi;
  int original_frame_width = -1;
  int original_frame_height = -1;
  bool init_roi = true;

  // TODO: remove these since they are only used for debugging
  int frame_count = 0;
  struct timeval stop, start, stop_all, start_all;

  while (true)
  {
    zmq::message_t request;
    sock.recv(request, zmq::recv_flags::none);
    std::string rpl = std::string(static_cast<char *>(request.data()), request.size());

    gettimeofday(&start_all, NULL);

    gettimeofday(&start, NULL);

    // decode image
    std::string dec_jpg = base64_decode(rpl);
    std::vector<uchar> data(dec_jpg.begin(), dec_jpg.end());
    cv::Mat rgb_image = cv::imdecode(cv::Mat(data), 1);
    cv::cvtColor(rgb_image, greyScale_image, cv::COLOR_BGR2GRAY);

    gettimeofday(&stop, NULL);
    printf("Decode: %f\n", (double) ((stop.tv_sec - start.tv_sec) * 1000000 + stop.tv_usec - start.tv_usec) / 1000000);

    if (original_frame_width == -1 && original_frame_height == -1)
    {
      original_frame_width = rgb_image.size().width;
      original_frame_height = rgb_image.size().height;
    }

    // Calculate ROI
    if (init_roi)
    {
      roi = detectSingleFace(rgb_image, face_model, det_parameters, greyScale_image);
      init_roi = false;
    }

    if (roi.width > 2 && rgb_image.size().width  == original_frame_width && rgb_image.size().height == original_frame_height)
    {
      // Crop the RGB and GrayScale frame based on ROI
      rgb_image_roi = rgb_image(roi);
      greyScale_image_roi = greyScale_image(roi);
    }
    else
    {
      rgb_image_roi = rgb_image;
      greyScale_image_roi = greyScale_image;
    }

    std::cout << "Image size: " << rgb_image.size() << " -> " << rgb_image_roi.size() << std::endl;

    gettimeofday(&start, NULL);

    // results will be stored in face_model
    bool landmark_detection_success = LandmarkDetector::DetectLandmarksInVideo(rgb_image_roi, face_model, det_parameters, greyScale_image_roi);

    gettimeofday(&stop, NULL);
    printf("DetectLandmarksInVideo: %f\n", (double) ((stop.tv_sec - start.tv_sec) * 1000000 + stop.tv_usec - start.tv_usec) / 1000000);

    string json;
    if (landmark_detection_success)
    {
      gettimeofday(&start, NULL);

      face_analyser.PredictStaticAUsAndComputeFeatures(rgb_image, face_model.detected_landmarks);

      auto aus_intensity = face_analyser.GetCurrentAUsReg();
      auto aus_presence = face_analyser.GetCurrentAUsClass();

      json = convertToJSON(roi, aus_intensity, aus_presence);

      gettimeofday(&stop, NULL);
      printf("PredictStaticAUsAndComputeFeatures & convertToJSON: %f\n", (double)((stop.tv_sec - start.tv_sec) * 1000000 + stop.tv_usec - start.tv_usec) / 1000000);
    }
    else
    {
      // Tracking or detection failed; so reset the model and set ROI to 0 in order to receive full frame from ZMQ
      if (
        (!face_model.tracking_initialised && (face_model.failures_in_a_row + 1) % (det_parameters.reinit_video_every * 6) == 0) 
        || (face_model.tracking_initialised && !face_model.detection_success && det_parameters.reinit_video_every > 0 && face_model.failures_in_a_row % det_parameters.reinit_video_every == 0))
      {
        std::cout << "RESET" << std::endl;

        init_roi = true;

        face_model.Reset();

        json = convertToJSON();
      } 
      else 
      {
        json = convertToJSON(roi);
      }
    }

    gettimeofday(&stop_all, NULL);
    printf("[%d] All: %f\n\n", frame_count, (double) ((stop_all.tv_sec - start_all.tv_sec) * 1000000 + stop_all.tv_usec - start_all.tv_usec) / 1000000);

    // Send reply back to client
    zmq::message_t reply(json.length());
    memcpy(reply.data(), json.c_str(), json.length());
    sock.send(reply, zmq::send_flags::none);

    // Save image for debugging purposes
		// for(int j = 0; j < face_model.detected_landmarks.rows / 2; j++){
		// 	float x = face_model.detected_landmarks[j][0];
		// 	float y = face_model.detected_landmarks[j + face_model.detected_landmarks.rows / 2][0];
		// 	cv::circle(rgb_image_roi, cv::Point2f(x,y), 4, cv::Scalar(255, 0, 0), cv::FILLED, cv::LINE_8);
		// }
		// cv::imwrite("../../experimental-hub-openface-test/" + to_string(frame_count) + ".jpg", rgb_image_roi);

    frame_count++;
  }

  return 0;
}

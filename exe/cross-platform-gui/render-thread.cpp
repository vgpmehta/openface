#include <iostream>
#include <unistd.h>

#include <QImage>

#include <GazeEstimation.h>
#include <LandmarkCoreIncludes.h>
#include <RecorderOpenFace.h>
#include <RecorderOpenFaceParameters.h>
#include <SequenceCapture.h>
#include <VisualizationUtils.h>
#include <Visualizer.h>
#include <FaceAnalyser.h>

#include "render-thread.h"

RenderThread::RenderThread(QObject *parent) : QThread(parent) {}

RenderThread::~RenderThread() {
  mutex.lock();
  abort = true;
  condition.wakeOne();
  mutex.unlock();
  wait();
}

void RenderThread::do_csv_work(void) {
  std::cout << "Slot called by timeout\n";
}


void RenderThread::run() {
  std::vector<std::string> arguments{"openface-cross-platform-gui", "-out_dir", "/Users/qz1llg/Repos/OpenFace/test-dump"};
  LandmarkDetector::FaceModelParameters det_parameters(arguments);

  // The modules that are being used for tracking
  LandmarkDetector::CLNF face_model(det_parameters.model_location);
  if (!face_model.loaded_successfully) {
    return;
  }

  if (!face_model.eye_model) {
  }

  // Open a sequence
  Utilities::SequenceCapture sequence_reader;

  FaceAnalysis::FaceAnalyserParameters face_analysis_params(arguments);
  FaceAnalysis::FaceAnalyser face_analyser(face_analysis_params);

  if (!face_model.eye_model) {
    std::cout << "WARNING: no eye model found\n";
  }

  if (face_analyser.GetAUClassNames().size() == 0 &&
      face_analyser.GetAUClassNames().size() == 0) {
    std::cout << "WARNING: no Action Unit models found\n";
  }

  // A utility for visualizing the results (show just the tracks)
  Utilities::Visualizer visualizer(true, false, false, false);

  // Tracking FPS for visualization
  Utilities::FpsTracker fps_tracker;
  fps_tracker.AddFrame();

  int sequence_number = 0;

  forever {
    if (!sequence_reader.OpenWebcam(0, 1224, 1284)) {
      break;
    }

    cv::Mat rgb_image = sequence_reader.GetNextFrame();

    Utilities::RecorderOpenFaceParameters recording_params{
        arguments,
        true,
        sequence_reader.IsWebcam(),
        sequence_reader.fx,
        sequence_reader.fy,
        sequence_reader.cx,
        sequence_reader.cy,
        sequence_reader.fps};

    Utilities::RecorderOpenFace open_face_rec(sequence_reader.name,
                                              recording_params, arguments);

    while (!rgb_image.empty()) {

      // Reading the images
      cv::Mat_<uchar> grayscale_image = sequence_reader.GetGrayFrame();

      // The actual facial landmark detection / tracking
      bool detection_success = LandmarkDetector::DetectLandmarksInVideo(
          rgb_image, face_model, det_parameters, grayscale_image);

      // Gaze tracking, absolute gaze direction
      cv::Point3f gazeDirection0(0, 0, -1);
      cv::Point3f gazeDirection1(0, 0, -1);
      cv::Vec2d gazeAngle(0, 0);

      // If tracking succeeded and we have an eye model, estimate gaze
      if (detection_success && face_model.eye_model) {
        GazeAnalysis::EstimateGaze(
            face_model, gazeDirection0, sequence_reader.fx, sequence_reader.fy,
            sequence_reader.cx, sequence_reader.cy, true);
        GazeAnalysis::EstimateGaze(
            face_model, gazeDirection1, sequence_reader.fx, sequence_reader.fy,
            sequence_reader.cx, sequence_reader.cy, false);
	gazeAngle = GazeAnalysis::GetGazeAngle(gazeDirection0, gazeDirection1);

      }
      cv::Mat sim_warped_img;
      cv::Mat_<double> hog_descriptor;
      int num_hog_rows = 0, num_hog_cols = 0;

      // Perform AU detection and HOG feature extraction, as this can be
      // expensive only compute it if needed by output or visualization
      if (recording_params.outputAlignedFaces() ||
          recording_params.outputHOG() || recording_params.outputAUs() ||
          visualizer.vis_align || visualizer.vis_hog || visualizer.vis_aus) {
        face_analyser.AddNextFrame(
            rgb_image, face_model.detected_landmarks,
            face_model.detection_success, sequence_reader.time_stamp,
            sequence_reader.IsWebcam());
        face_analyser.GetLatestAlignedFace(sim_warped_img);
        face_analyser.GetLatestHOG(hog_descriptor, num_hog_rows, num_hog_cols);
      }

      // Work out the pose of the head from the tracked model
      cv::Vec6d pose_estimate = LandmarkDetector::GetPose(
          face_model, sequence_reader.fx, sequence_reader.fy,
          sequence_reader.cx, sequence_reader.cy);

      // Keeping track of FPS
      fps_tracker.AddFrame();

      // // Displaying the tracking visualizations
      visualizer.SetImage(rgb_image, sequence_reader.fx, sequence_reader.fy,
                          sequence_reader.cx, sequence_reader.cy);
      visualizer.SetObservationLandmarks(face_model.detected_landmarks,
                                         face_model.detection_certainty,
                                         face_model.GetVisibilities());
      visualizer.SetObservationPose(pose_estimate,
                                    face_model.detection_certainty);
      visualizer.SetObservationGaze(
          gazeDirection0, gazeDirection1,
          LandmarkDetector::CalculateAllEyeLandmarks(face_model),
          LandmarkDetector::Calculate3DEyeLandmarks(
              face_model, sequence_reader.fx, sequence_reader.fy,
              sequence_reader.cx, sequence_reader.cy),
          face_model.detection_certainty);
      visualizer.SetFps(fps_tracker.GetFPS());

      rgb_image = visualizer.GetVisImage();

      open_face_rec.SetObservationHOG(detection_success, hog_descriptor,
                                      num_hog_rows, num_hog_cols,
                                      31); // The number of channels in HOG is
                                           // fixed at the moment, as using FHOG
      open_face_rec.SetObservationVisualization(visualizer.GetVisImage());
      open_face_rec.SetObservationActionUnits(
          face_analyser.GetCurrentAUsReg(), face_analyser.GetCurrentAUsClass());
      open_face_rec.SetObservationLandmarks(
          face_model.detected_landmarks,
          face_model.GetShape(sequence_reader.fx, sequence_reader.fy,
                              sequence_reader.cx, sequence_reader.cy),
          face_model.params_global, face_model.params_local,
          face_model.detection_certainty, detection_success);
      open_face_rec.SetObservationPose(pose_estimate);
      open_face_rec.SetObservationGaze(
          gazeDirection0, gazeDirection1, gazeAngle,
          LandmarkDetector::CalculateAllEyeLandmarks(face_model),
          LandmarkDetector::Calculate3DEyeLandmarks(
              face_model, sequence_reader.fx, sequence_reader.fy,
              sequence_reader.cx, sequence_reader.cy));
      open_face_rec.SetObservationTimestamp(sequence_reader.time_stamp);
      open_face_rec.SetObservationFaceID(0);
      open_face_rec.SetObservationFrameNumber(sequence_reader.GetFrameNumber());
      open_face_rec.SetObservationFaceAlign(sim_warped_img);
      open_face_rec.WriteObservation();
      open_face_rec.WriteObservationTracked();

      // face_analyser.PostprocessOutputFile(open_face_rec.GetCSVFile());

      copyMakeBorder(rgb_image, rgb_image, rgb_image.rows / 7,
                     rgb_image.rows / 7, rgb_image.cols / 7, rgb_image.cols / 7,
                     cv::BORDER_WRAP);

      QImage qimg(rgb_image.data, rgb_image.cols, rgb_image.rows,
                  rgb_image.step, QImage::Format_RGB888);

      double scaleFactor = 2;
      emit renderedImage(qimg, scaleFactor);

      // Grabbing the next frame in the sequence
      rgb_image = sequence_reader.GetNextFrame();
    }
    // Reset the model, for the next video
    face_model.Reset();
    sequence_reader.Close();
    sequence_number++;
  }
}

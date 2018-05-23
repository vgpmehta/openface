#include <iostream>
#include <unistd.h>

#include <QImage>

#include <GazeEstimation.h>
#include <LandmarkCoreIncludes.h>
#include <SequenceCapture.h>
#include <VisualizationUtils.h>
#include <Visualizer.h>

#include "render-thread.h"

RenderThread::RenderThread(QObject *parent) : QThread(parent) {}

RenderThread::~RenderThread() {
  mutex.lock();
  abort = true;
  condition.wakeOne();
  mutex.unlock();
  wait();
}

void RenderThread::run() {
  std::vector<std::string> arguments{"openface-cross-platform-gui"};
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

  // A utility for visualizing the results (show just the tracks)
  Utilities::Visualizer visualizer(true, false, false, false);

  // Tracking FPS for visualization
  Utilities::FpsTracker fps_tracker;
  fps_tracker.AddFrame();

  int sequence_number = 0;

  forever {
    // this is not a for loop as we might also be reading from a webcam
    while (true) {
      if (!sequence_reader.OpenWebcam(0, 1224, 1284)) {
        break;
      }

      cv::Mat rgb_image = sequence_reader.GetNextFrame();

      while (!rgb_image.empty()) {

        // Reading the images
        cv::Mat_<uchar> grayscale_image = sequence_reader.GetGrayFrame();

        // The actual facial landmark detection / tracking
        bool detection_success = LandmarkDetector::DetectLandmarksInVideo(
            rgb_image, face_model, det_parameters, grayscale_image);

        // Gaze tracking, absolute gaze direction
        cv::Point3f gazeDirection0(0, 0, -1);
        cv::Point3f gazeDirection1(0, 0, -1);

        // If tracking succeeded and we have an eye model, estimate gaze
        if (detection_success && face_model.eye_model) {
          GazeAnalysis::EstimateGaze(
              face_model, gazeDirection0, sequence_reader.fx,
              sequence_reader.fy, sequence_reader.cx, sequence_reader.cy, true);
          GazeAnalysis::EstimateGaze(face_model, gazeDirection1,
                                     sequence_reader.fx, sequence_reader.fy,
                                     sequence_reader.cx, sequence_reader.cy,
                                     false);
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

        copyMakeBorder(rgb_image, rgb_image, rgb_image.rows / 7,
                       rgb_image.rows / 7, rgb_image.cols / 7,
                       rgb_image.cols / 7, cv::BORDER_WRAP);

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
}

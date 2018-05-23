// -*- c++ -*-
#include <iostream>
#include <unistd.h>
#include <vector>


#include <QApplication>
#include <QThread>
#include <QWidget>

#include <GazeEstimation.h>
#include <LandmarkCoreIncludes.h>
#include <SequenceCapture.h>
#include <VisualizationUtils.h>
#include <Visualizer.h>

class gui_application : public QWidget {
public:
  void start(void) {
    resize(450, 350);
    setWindowTitle("OpenFace Demo");
    show();
  }
};


class worker {
public:
  worker() { }
  ~worker() { }
  void start(QWidget &application) {

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
        // Grabbing the next frame in the sequence
        rgb_image = sequence_reader.GetNextFrame();
	std::cout << rgb_image;
      }
      // Reset the model, for the next video
      face_model.Reset();
      sequence_reader.Close();
      sequence_number++;
    }
  }
};

int main(int argc, char **argv) {
  QApplication handle{argc, argv};

  gui_application gui{};

  QThread *thread = QThread::create([&gui] {
      worker image_processing_worker{};
      image_processing_worker.start(gui);
    });

  thread->setObjectName("VideoProcessingWorkerOpenFace");
  thread->start();
  gui.start();

  return handle.exec();
}

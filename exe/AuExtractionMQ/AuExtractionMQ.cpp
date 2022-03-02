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
//       Tadas Baltrušaitis, Amir Zadeh, Yao Chong Lim, and Louis-Philippe Morency
//       in IEEE International Conference on Automatic Face and Gesture Recognition, 2018
//
//       Convolutional experts constrained local model for facial landmark detection.
//       A. Zadeh, T. Baltrušaitis, and Louis-Philippe Morency,
//       in Computer Vision and Pattern Recognition Workshops, 2017.
//
//       Rendering of Eyes for Eye-Shape Registration and Gaze Estimation
//       Erroll Wood, Tadas Baltrušaitis, Xucong Zhang, Yusuke Sugano, Peter Robinson, and Andreas Bulling
//       in IEEE International. Conference on Computer Vision (ICCV),  2015
//
//       Cross-dataset learning and person-specific normalisation for automatic Action Unit detection
//       Tadas Baltrušaitis, Marwa Mahmoud, and Peter Robinson
//       in Facial Expression Recognition and Analysis Challenge,
//       IEEE International Conference on Automatic Face and Gesture Recognition, 2015
//
///////////////////////////////////////////////////////////////////////////////


// AuExtractionMQ.cpp : Defines the entry point for the Action unit extraction console application with a stream to ZeroMQ.

// Local includes
#include "LandmarkCoreIncludes.h"
#include <Face_utils.h>
#include <FaceAnalyser.h>
#include <GazeEstimation.h>
#include <RecorderOpenFace.h>
#include <RecorderOpenFaceParameters.h>
#include <SequenceCapture.h>
#include <Visualizer.h>
#include <VisualizationUtils.h>
#include <chrono>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <string>
#define ZMQ_STATIC
#include <zmq.hpp>

#ifndef CONFIG_DIR
#define CONFIG_DIR "~"
#endif

#define INFO_STREAM( stream ) \
std::cout << stream << std::endl

#define WARN_STREAM( stream ) \
std::cout << "Warning: " << stream << std::endl

#define ERROR_STREAM( stream ) \
std::cout << "Error: " << stream << std::endl

using namespace std::chrono;

static void printErrorAndAbort(const std::string & error)
{
	std::cout << error << std::endl;
}

#define FATAL_STREAM( stream ) \
printErrorAndAbort( std::string( "Fatal error: " ) + stream )

std::vector<std::string> get_arguments(int argc, char **argv)
{

	std::vector<std::string> arguments;

	// First argument is reserved for the name of the executable
	for (int i = 0; i < argc; ++i)
	{
		arguments.push_back(std::string(argv[i]));
	}
	return arguments;
}

int main(int argc, char **argv)
{

	std::vector<std::string> arguments = get_arguments(argc, argv);

	// no arguments: output usage
	if (arguments.size() == 1)
	{
		std::cout << "For command line arguments see:" << std::endl;
		std::cout << " https://github.com/TadasBaltrusaitis/OpenFace/wiki/Command-line-arguments";
		return 0;
	}

	bool side_side_h = false;
	bool side_side_v = false;
	int zmq_port = 5556;
	for (size_t i = 1; i < arguments.size(); ++i)
	{

		if (arguments[i].compare("-side_side_h") == 0)
		{
			side_side_h = true;
		}
		else if (arguments[i].compare("-side_side_v") == 0)
		{
			side_side_v = true;
		}
		else if (arguments[i].compare("-zmq_port") == 0)
		{
			zmq_port = stoi(arguments[i + 1]);
		}
	}

	// Load the modules that are being used for tracking and face analysis
	// Load face landmark detector
	LandmarkDetector::FaceModelParameters det_parameters(arguments);
	// Always track gaze in feature extraction
	LandmarkDetector::CLNF face_model(det_parameters.model_location);

	if (!face_model.loaded_successfully)
	{
		std::cout << "ERROR: Could not load the landmark detector" << std::endl;
		return 1;
	}

	// Load facial feature extractor and AU analyser
	FaceAnalysis::FaceAnalyserParameters face_analysis_params(arguments);
	FaceAnalysis::FaceAnalyser face_analyser(face_analysis_params);

	if (!face_model.eye_model)
	{
		std::cout << "WARNING: no eye model found" << std::endl;
	}

	if (face_analyser.GetAUClassNames().size() == 0 && face_analyser.GetAUClassNames().size() == 0)
	{
		std::cout << "WARNING: no Action Unit models found" << std::endl;
	}

	Utilities::SequenceCapture sequence_reader;

	// A utility for visualizing the results
	Utilities::Visualizer visualizer(arguments);

	// Tracking FPS for visualization
	Utilities::FpsTracker fps_tracker;
	fps_tracker.AddFrame();


	//  Prepare our context and socket
	zmq::context_t context(1);
	zmq::socket_t sock(context, ZMQ_PUB);
	std::cout << "prev sock bind" << std::endl;

	std::stringstream zmq_ss0;
	zmq_ss0 << "tcp://*:" << zmq_port;
	std::string zmq_connect_string0 = zmq_ss0.str();
	std::cout << "ZMQ connect : " << zmq_connect_string0 << std::endl;

	sock.bind(zmq_connect_string0.c_str());
	std::cout << "sock bind" << std::endl;
	std::stringstream zmq_ss;
	zmq_ss << "tcp://localhost:" << zmq_port;
	std::string zmq_connect_string = zmq_ss.str();
	std::cout << "ZMQ connect : " << zmq_connect_string << std::endl;
	int sock_connected;
	sock_connected  = zmq_connect(sock, zmq_connect_string.c_str());
	std::cout << "sock connected" << sock_connected <<std::endl;

	while (true) // this is not a for loop as we might also be reading from a webcam
	{

		// The sequence reader chooses what to open based on command line arguments provided
		if (!sequence_reader.Open(arguments))
			break;

		INFO_STREAM("Device or file opened");

		if (sequence_reader.IsWebcam())
		{
			INFO_STREAM("WARNING: using a webcam in feature extraction, Action Unit predictions will not be as accurate in real-time webcam mode");
			INFO_STREAM("WARNING: using a webcam in feature extraction, forcing visualization of tracking to allow quitting the application (press q)");
			visualizer.vis_track = true;
		}

		cv::Mat captured_image;

		Utilities::RecorderOpenFaceParameters recording_params(arguments, true, sequence_reader.IsWebcam(),
			sequence_reader.fx, sequence_reader.fy, sequence_reader.cx, sequence_reader.cy, sequence_reader.fps);
		if (!face_model.eye_model)
		{
			recording_params.setOutputGaze(false);
		}
		Utilities::RecorderOpenFace open_face_rec(sequence_reader.name, recording_params, arguments);

		if (recording_params.outputGaze() && !face_model.eye_model)
			std::cout << "WARNING: no eye model defined, but outputting gaze" << std::endl;

		captured_image = sequence_reader.GetNextFrame();

		// For reporting progress
		double reported_completion = 0;

		INFO_STREAM("Starting tracking");
		while (!captured_image.empty())
		{
			// Converting to grayscale
			cv::Mat_<uchar> grayscale_image = sequence_reader.GetGrayFrame();


			// The actual facial landmark detection / tracking
			bool detection_success = LandmarkDetector::DetectLandmarksInVideo(captured_image, face_model, det_parameters, grayscale_image);

			// Gaze tracking, absolute gaze direction
			cv::Point3f gazeDirection0(0, 0, 0); cv::Point3f gazeDirection1(0, 0, 0); cv::Vec2d gazeAngle(0, 0);

			if (detection_success && face_model.eye_model)
			{
				GazeAnalysis::EstimateGaze(face_model, gazeDirection0, sequence_reader.fx, sequence_reader.fy, sequence_reader.cx, sequence_reader.cy, true);
				GazeAnalysis::EstimateGaze(face_model, gazeDirection1, sequence_reader.fx, sequence_reader.fy, sequence_reader.cx, sequence_reader.cy, false);
				gazeAngle = GazeAnalysis::GetGazeAngle(gazeDirection0, gazeDirection1);
			}

			// Do face alignment
			cv::Mat sim_warped_img;
			cv::Mat_<double> hog_descriptor; int num_hog_rows = 0, num_hog_cols = 0;

			// Perform AU detection and HOG feature extraction, as this can be expensive only compute it if needed by output or visualization
			if (recording_params.outputAlignedFaces() || recording_params.outputHOG() || recording_params.outputAUs() || visualizer.vis_align || visualizer.vis_hog || visualizer.vis_aus)
			{
				face_analyser.AddNextFrame(captured_image, face_model.detected_landmarks, face_model.detection_success, sequence_reader.time_stamp, sequence_reader.IsWebcam());
				face_analyser.GetLatestAlignedFace(sim_warped_img);
				face_analyser.GetLatestHOG(hog_descriptor, num_hog_rows, num_hog_cols);
			}

			// Work out the pose of the head from the tracked model
			cv::Vec6d pose_estimate = LandmarkDetector::GetPose(face_model, sequence_reader.fx, sequence_reader.fy, sequence_reader.cx, sequence_reader.cy);

			// Keeping track of FPS
			fps_tracker.AddFrame();
			
			//get timestamp
			// get current time
			auto now = system_clock::now();

			// get number of milliseconds for the current second
			// (remainder after division into seconds)
			auto ms = duration_cast<milliseconds>(now.time_since_epoch()) % 1000;

			// convert to std::time_t in order to convert to std::tm (broken time)
			auto timer = system_clock::to_time_t(now);

			// convert to broken time
			std::tm bt = *std::localtime(&timer);

			std::ostringstream ss;

			ss << "{ 'timestamp' : '" << std::put_time(&bt, "%H:%M:%S"); // HH:MM:SS
			ss << '.' << std::setfill('0') << std::setw(3) << ms.count() ;
			// AUs
			/*std::vector<std::pair<std::string, double>> aus_pres_vec = face_analyser.GetCurrentAUsClass();
			std::vector<std::pair<std::string, double>> aus_intensity_vec = face_analyser.GetCurrentAUsReg();*/
			// postProcess 
			// postProcess 
			std::pair < std::vector<std::pair<std::string, std::vector<double>>>, std::vector<std::pair<std::string, std::vector<double>>>>AUs;
			AUs = face_analyser.Live_PostprocessOutputFile();
			std::vector<std::pair<std::string, std::vector<double>>> aus_reg = AUs.first;
			std::vector<std::pair<std::string, std::vector<double>>> aus_class = AUs.second;

			for (size_t i = 0; i < aus_reg.size(); ++i)
			{
				ss << "' , '" << aus_class[i].first << "_presence' : '" << aus_class[i].second.back() <<"' , '" << aus_reg[i].first << "_intensity' : '" << aus_reg[i].second.back();
			}
			// Gaze
			ss << "' , 'gazeDirection0' : '" << gazeDirection0 << "' , 'gazeDirection1' : '" << gazeDirection1;
			// HeadPose [Tx, Ty, Tz, Eul_x, Eul_y, Eul_z]
			ss << "' , 'Tx' : '" << pose_estimate[0] << "' , 'Ty' : '" << pose_estimate[1] << "' , 'Tz' : '" << pose_estimate[2] << "' , 'Eul_x' : '" << pose_estimate[3] << "' , 'Eul_y' : '" << pose_estimate[4] << "' , 'Eul_z' : '" << pose_estimate[5];
			ss << "' } ";
			std::string aus_str = ss.str();
			
			std::cout << aus_str  << std::endl;
			zmq::message_t reply(aus_str.size());
			memcpy(reply.data(), aus_str.c_str(), aus_str.size());
			sock.send(reply, zmq::send_flags::none);
			
			//// Setting up the recorder output
			//std::cout << "open_face_rec 0" << std::endl;
			//open_face_rec.SetObservationActionUnits(face_analyser.GetCurrentAUsReg(), face_analyser.GetCurrentAUsClass());
			//std::cout << "open_face_rec 1" << std::endl;
			//open_face_rec.SetObservationPose(pose_estimate);
			//std::cout << "open_face_rec 2" << std::endl;
			//open_face_rec.SetObservationGaze(gazeDirection0, gazeDirection1, gazeAngle, LandmarkDetector::CalculateAllEyeLandmarks(face_model), LandmarkDetector::Calculate3DEyeLandmarks(face_model, sequence_reader.fx, sequence_reader.fy, sequence_reader.cx, sequence_reader.cy));
			//std::cout << "open_face_rec 3" << std::endl;
			//open_face_rec.SetObservationTimestamp(sequence_reader.time_stamp);
			//std::cout << "open_face_rec 4" << std::endl;
			//open_face_rec.SetObservationFaceID(0);
			//std::cout << "open_face_rec 5" << std::endl;
			//open_face_rec.SetObservationFrameNumber(sequence_reader.GetFrameNumber());
			//std::cout << "open_face_rec 6" << std::endl;
			//open_face_rec.WriteObservation();
			//std::cout << "open_face_rec 7" << std::endl;
			//// Reporting progress
			//if (sequence_reader.GetProgress() >= reported_completion / 10.0)
			//{
			//	std::cout << reported_completion * 10 << "% ";
			//	if (reported_completion == 10)
			//	{
			//		std::cout << std::endl;
			//	}
			//	reported_completion = reported_completion + 1;
			//}

			// Grabbing the next frame in the sequence
			captured_image = sequence_reader.GetNextFrame();

		}

		INFO_STREAM("Closing output recorder");
		open_face_rec.Close();
		INFO_STREAM("Closing input reader");
		sequence_reader.Close();
		INFO_STREAM("Closed successfully");

		/*if (recording_params.outputAUs())
		{
			INFO_STREAM("Postprocessing the Action Unit predictions");
			face_analyser.PostprocessOutputFile(open_face_rec.GetCSVFile());
		}*/

		// Reset the models for the next video
		face_analyser.Reset();
		face_model.Reset();

	}

	return 0;
}

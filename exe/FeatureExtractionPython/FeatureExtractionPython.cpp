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

// tuned by frankiezafe, http://polymorph.cool

// FeatureExtraction.cpp : Defines the entry point for the feature extraction console application.

#include <boost/python.hpp>
#include <boost/smart_ptr.hpp>
#include <boost/thread.hpp>
#include <boost/date_time.hpp>
#include <boost/function.hpp>
#include <boost/bind.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <unistd.h>

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

#ifndef CONFIG_DIR
#define CONFIG_DIR "~"
#endif

#define INFO_STREAM( stream ) \
std::cout << stream << std::endl

#define WARN_STREAM( stream ) \
std::cout << "Warning: " << stream << std::endl

#define ERROR_STREAM( stream ) \
std::cout << "Error: " << stream << std::endl

static void printErrorAndAbort(const std::string & error)
{
	std::cout << error << std::endl;
}

#define FATAL_STREAM( stream ) \
printErrorAndAbort( std::string( "Fatal error: " ) + stream )

#define LANDMARKS_NUM 68

boost::shared_mutex _access;

int OpenfaceVideoWorker( 
	std::vector<std::string>& args, 
	const bool& running,
	boost::function<void(Utilities::RecorderOpenFace&)> callback,
	boost::thread_group & thread_group, 
	boost::thread * this_thread ) {
    
	// Load the modules that are being used for tracking and face analysis
	// Load face landmark detector
	LandmarkDetector::FaceModelParameters det_parameters(args);
	// Always track gaze in feature extraction
	LandmarkDetector::CLNF face_model(det_parameters.model_location);

	if (!face_model.loaded_successfully)
	{
		std::cout << "ERROR: Could not load the landmark detector" << std::endl;
		return 1;
	}

	// Load facial feature extractor and AU analyser
	FaceAnalysis::FaceAnalyserParameters face_analysis_params(args);
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
	Utilities::Visualizer visualizer(args);

	// Tracking FPS for visualization
	Utilities::FpsTracker fps_tracker;
	fps_tracker.AddFrame();

	while ( running ) // this is not a for loop as we might also be reading from a webcam
	{

		// The sequence reader chooses what to open based on command line arguments provided
		if (!sequence_reader.Open(args))
			break;

		INFO_STREAM("Device or file opened");

		if (sequence_reader.IsWebcam())
		{
			INFO_STREAM("WARNING: using a webcam in feature extraction, Action Unit predictions will not be as accurate in real-time webcam mode");
			INFO_STREAM("WARNING: using a webcam in feature extraction, forcing visualization of tracking to allow quitting the application (press q)");
			visualizer.vis_track = true;
		}

		cv::Mat captured_image;

		Utilities::RecorderOpenFaceParameters recording_params(args, true, sequence_reader.IsWebcam(),
			sequence_reader.fx, sequence_reader.fy, sequence_reader.cx, sequence_reader.cy, sequence_reader.fps);
		if (!face_model.eye_model)
		{
			recording_params.setOutputGaze(false);
		}
		
		Utilities::RecorderOpenFace open_face_rec(sequence_reader.name, recording_params, args);

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

			// Displaying the tracking visualizations
			visualizer.SetImage(captured_image, sequence_reader.fx, sequence_reader.fy, sequence_reader.cx, sequence_reader.cy);
			visualizer.SetObservationFaceAlign(sim_warped_img);
			visualizer.SetObservationHOG(hog_descriptor, num_hog_rows, num_hog_cols);
			visualizer.SetObservationLandmarks(face_model.detected_landmarks, face_model.detection_certainty, face_model.GetVisibilities());
			visualizer.SetObservationPose(pose_estimate, face_model.detection_certainty);
			visualizer.SetObservationGaze(gazeDirection0, gazeDirection1, LandmarkDetector::CalculateAllEyeLandmarks(face_model), LandmarkDetector::Calculate3DEyeLandmarks(face_model, sequence_reader.fx, sequence_reader.fy, sequence_reader.cx, sequence_reader.cy), face_model.detection_certainty);
			visualizer.SetObservationActionUnits(face_analyser.GetCurrentAUsReg(), face_analyser.GetCurrentAUsClass());
			visualizer.SetFps(fps_tracker.GetFPS());

			// detect key presses
			char character_press = visualizer.ShowObservation();
			
			// quit processing the current sequence (useful when in Webcam mode)
			if (character_press == 'q')
			{
				break;
			}

			// Setting up the recorder output
			open_face_rec.SetObservationHOG(detection_success, hog_descriptor, num_hog_rows, num_hog_cols, 31); // The number of channels in HOG is fixed at the moment, as using FHOG
			open_face_rec.SetObservationVisualization(visualizer.GetVisImage());
			open_face_rec.SetObservationActionUnits(face_analyser.GetCurrentAUsReg(), face_analyser.GetCurrentAUsClass());
			open_face_rec.SetObservationLandmarks(face_model.detected_landmarks, face_model.GetShape(sequence_reader.fx, sequence_reader.fy, sequence_reader.cx, sequence_reader.cy),
				face_model.params_global, face_model.params_local, face_model.detection_certainty, detection_success);
			open_face_rec.SetObservationPose(pose_estimate);
			open_face_rec.SetObservationGaze(gazeDirection0, gazeDirection1, gazeAngle, LandmarkDetector::CalculateAllEyeLandmarks(face_model), LandmarkDetector::Calculate3DEyeLandmarks(face_model, sequence_reader.fx, sequence_reader.fy, sequence_reader.cx, sequence_reader.cy));
			open_face_rec.SetObservationTimestamp(sequence_reader.time_stamp);
			open_face_rec.SetObservationFaceID(0);
			open_face_rec.SetObservationFrameNumber(sequence_reader.GetFrameNumber());
			open_face_rec.SetObservationFaceAlign(sim_warped_img);
			open_face_rec.WriteObservation();
			open_face_rec.WriteObservationTracked();

			// CALLING OPENVIDEO CALLBACK
			callback( open_face_rec );
			
			// Reporting progress
			if (sequence_reader.GetProgress() >= reported_completion / 10.0)
			{
				std::cout << reported_completion * 10 << "% ";
				if (reported_completion == 10)
				{
					std::cout << std::endl;
				}
				reported_completion = reported_completion + 1;
			}

			// Grabbing the next frame in the sequence
			captured_image = sequence_reader.GetNextFrame();

		}

		INFO_STREAM("Closing output recorder");
		open_face_rec.Close();
		INFO_STREAM("Closing input reader");
		sequence_reader.Close();
		INFO_STREAM("Closed successfully");

		if (recording_params.outputAUs())
		{
			INFO_STREAM("Postprocessing the Action Unit predictions");
			face_analyser.PostprocessOutputFile(open_face_rec.GetCSVFile());
		}

		// Reset the models for the next video
		face_analyser.Reset();
		face_model.Reset();
		
		thread_group.remove_thread(this_thread);
		delete this_thread;

	}
	
	return 0;
	
}

struct OpenfaceV3 {
	
	float x;
	float y;
	float z;
	
	OpenfaceV3() : x(0), y(0), z(0) {}
	
	OpenfaceV3( const OpenfaceV3& src ) {
		(*this) = src;
	}
	
	~OpenfaceV3() {}
	
	bool operator == ( const OpenfaceV3& src ) const {
		return x == src.x && y == src.y && z == src.z;
	}
	
	bool operator != ( const OpenfaceV3& src ) const {
		return x != src.x || y != src.y || z != src.z;
	}
	
	void operator = ( const OpenfaceV3& src ) {
		x = src.x;
		y = src.y;
		z = src.z;
	}
	
	void operator = ( const cv::Point3f& src ) {
		x = src.x;
		y = src.y;
		z = src.z;
	}
	
	void set( const float& vx, const float& vy, const float& vz ) {
		x = vx;
		y = vy;
		z = vz;
	}
	
};

typedef std::vector<OpenfaceV3> OpenfaceV3Vector;

struct OpenfaceFrame {
	
	uint32_t fID;
	bool updated;
	OpenfaceV3 gaze_0;
	OpenfaceV3 gaze_1;
	OpenfaceV3 head_rot;
	OpenfaceV3 head_pos;
	std::vector<OpenfaceV3> landmarks;
	
	OpenfaceFrame(): updated(false), fID(0) {
		landmarks.resize(LANDMARKS_NUM);
	}
	
	OpenfaceFrame( const OpenfaceFrame& src ) {
		(*this) = src;
	}
	
	~OpenfaceFrame() {}
	
	void print() {
		std::cout <<
			"OpenfaceFrame" << std::endl <<
			"\t" << "fID: " << fID << std::endl <<
			"\t" << "updated: " << updated << std::endl <<
			"\t" << "gaze_0: " << gaze_0.x << ", "  << gaze_0.y << ", "  << gaze_0.z << std::endl <<
			"\t" << "gaze_1: " << gaze_1.x << ", "  << gaze_1.y << ", "  << gaze_1.z << std::endl <<
			"\t" << "head_rot: " << head_rot.x << ", "  << head_rot.y << ", "  << head_rot.z << std::endl <<
			"\t" << "head_pos: " << head_pos.x << ", "  << head_pos.y << ", "  << head_pos.z << std::endl;
		for ( int i = 0; i < LANDMARKS_NUM; ++i ) {
			std::cout << "\tlandmark [" << i << "] : " <<
				landmarks[i].x << ", "  << landmarks[i].y << ", "  << landmarks[i].z << std::endl;
		}
	}
	
	bool operator == ( const OpenfaceFrame& src ) {
		if ( 
			fID != src.fID || 
			gaze_0 != src.gaze_0 || 
			gaze_1 != src.gaze_1 || 
			head_rot != src.head_rot  || 
			head_pos != src.head_pos
		) { 
			return false;
		}
		for ( int i = 0; i < LANDMARKS_NUM; ++i ) {
			if ( landmarks[i] != src.landmarks[i] ) { return false; }
		}
		return true;
	}
	
	bool operator != ( const OpenfaceFrame& src ) {
		if ( 
			fID != src.fID || 
			gaze_0 != src.gaze_0 || 
			gaze_1 != src.gaze_1 || 
			head_rot != src.head_rot  || 
			head_pos != src.head_pos
		) { 
			return true;
		}
		for ( int i = 0; i < LANDMARKS_NUM; ++i ) {
			if ( landmarks[i] != src.landmarks[i] ) { return true; }
		}
		return false;
	}
	
	void operator = ( const OpenfaceFrame& src ) {
		fID = src.fID;
		updated = src.updated;
		gaze_0 = src.gaze_0;
		gaze_1 = src.gaze_1;
		head_rot = src.head_rot;
		head_pos = src.head_pos;
		for ( int i = 0; i < LANDMARKS_NUM; ++i ) {
			landmarks[i] = src.landmarks[i];
		}
	}

};

class OpenfaceVideo {
	
public:
	
	OpenfaceFrame public_frame;
	
	OpenfaceVideo() : worker(0) {}
	
	OpenfaceVideo( const OpenfaceVideo& src ) : worker(0) {
		std::cout << "OpenfaceVideo copy cstr is not implemented" << std::endl;
	}
	
	~OpenfaceVideo() {
		// clean stop
		stop();
	}
	
	void load_arguments(int argc, char **argv) {
		for (int i = 0; i < argc; ++i) {
			arguments.push_back(std::string(argv[i]));
		}
		//append_argument( "-2Dfp", "" );
		append_argument( "-3Dfp", "", false );
		append_argument( "-pose", "", false );
		append_argument( "-gaze", "", false );
		append_argument( "-device", "0", false );
	}

	int get_device() {
		return atoi( get_argument( "-device" ).c_str() );
	}
	
	void set_device( int d ) {
		stop();
		append_argument( "-device", std::to_string(d), true );
	}
	
	std::string get_landmark_model() {
		return get_argument( "-mloc" );
	}
	
	void set_landmark_model( std::string path ) {
		append_argument( "-mloc", path, true );
	}
	
	std::string get_HAAR() {
		return get_argument( "-fdloc" );
	}
	
	void set_HAAR( const char* path ) {
		append_argument( "-fdloc", std::string( path ), true );
	}
	
	std::string get_MTCNN() {
		return get_argument( "-mtcnnloc" );
	}
	
	void set_MTCNN( const char* path ) {
		append_argument( "-mtcnnloc", std::string( path ), true );
	}
	
	bool start() {
		
		if ( arguments.empty() ) {
			load_arguments(0,0);
		}
		
		stop();
		thread_running = true;		
		worker = new boost::thread();
		boost::function<void(Utilities::RecorderOpenFace&)> callback = boost::bind( &OpenfaceVideo::new_frame, this, _1 );
		*worker = boost::thread(
			boost::bind(
				&OpenfaceVideoWorker,
				arguments, 
				thread_running, 
				callback,
				boost::ref(openfacevideo_threads), 
				worker
			)
		);
		openfacevideo_threads.add_thread(worker);
		
		return true;
		
	}
	
	void new_frame( Utilities::RecorderOpenFace& rec ) {
		
		boost::unique_lock< boost::shared_mutex > lock(_access);
		
		const cv::Mat_<float>& l3d = rec.get_landmarks_3D();
		cv::Size s = l3d.size();
		
		frame.gaze_0 = rec.get_gaze_direction(0);
		frame.gaze_1 = rec.get_gaze_direction(1);
		
		cv::Vec6f h = rec.get_head_pose();
		frame.head_pos.set( h[0], h[1], h[2] );
		frame.head_rot.set( h[3], h[4], h[5] );
		
		for ( int c = 0;  c < s.width; ++c ) {
			frame.landmarks[c].x = l3d.at<float>( 0,c );
			frame.landmarks[c].y = l3d.at<float>( 1,c );
			frame.landmarks[c].z = l3d.at<float>( 2,c );
		}
		frame.fID++;
		frame.updated = true;
		
	}
	
	bool stop() {
		if ( thread_running ) {
			thread_running = false;
			usleep( 0.1 );
			worker = 0;
			return true;
		}
		return false;
	}
	
	bool is_running() const {
		return worker != 0;
	}
		
	bool has_frame() {
		boost::unique_lock< boost::shared_mutex > lock(_access);
		if ( frame.updated ) {
			public_frame = frame;
			frame.updated = false;
			return true;
		}
		return false;
	}
	
	OpenfaceFrame get_frame() {
		boost::unique_lock< boost::shared_mutex > lock(_access);
		frame.updated = false;
		return frame;
	}

private:
	
	void append_argument( std::string a, std::string value, bool empty_test ) {
		
		if ( empty_test && arguments.empty() ) {
			load_arguments(0,0);
		}
		
		std::vector<std::string>::iterator it = arguments.begin();
		std::vector<std::string>::iterator ite = arguments.end();
		bool found = false;
		for ( ; it != ite; ++it ) {
			if ( (*it).compare( a ) == 0 ) {
				++it;
				if ( value.length() > 0 || it == ite ) {
					std::cout << "arg " << a << " is already at " << value << std::endl;
					found = true; break;
				} else {
					(*it) = value;
					std::cout << "arg " << a << " > " << value << std::endl;
					found = true; break;
				}
			}
		}
		if ( !found ) {
			arguments.push_back( a );
			if (value.length() > 0) {
				arguments.push_back( value );
			}
		}
	}
	
	std::string get_argument( std::string a ) {
		std::vector<std::string>::iterator it = arguments.begin();
		std::vector<std::string>::iterator ite = arguments.end();
		for ( ; it != ite; ++it ) {
			if ( (*it).compare( a ) == 0 ) {
				++it;
				if ( it == ite ) {
					return "";
				}
				return (*it);
			}
		}
		return "";
	}
	
	std::vector<std::string> arguments;
	bool thread_running;
	bool _data_updated;
	boost::thread_group openfacevideo_threads;
	boost::thread* worker;
	
	OpenfaceFrame frame;
	
};

class with_gil {
public:
	with_gil()  { state_ = PyGILState_Ensure(); }
	~with_gil() { PyGILState_Release(state_);   }
	with_gil(const with_gil&)            = delete;
	with_gil& operator=(const with_gil&) = delete;
private:
	PyGILState_STATE state_;
};

class py_callable {
public:
	
	/// @brief Constructor that assumes the caller has the GIL locked.
	py_callable(const boost::python::object& object) {
		with_gil gil;
		object_.reset(
			// GIL locked, so it is safe to copy.
			new boost::python::object{object},
			// Use a custom deleter to hold GIL when the object is deleted.
			[](boost::python::object* object) {
				with_gil gil;
				delete object;
			});
	}

	// Use default copy-constructor and assignment-operator.
	py_callable(const py_callable&) = default;
	py_callable& operator=(const py_callable&) = default;

	template <typename ...Args>
	void operator()(Args... args) {
		// Lock the GIL as the python object is going to be invoked.
		with_gil gil;
		(*object_)(std::forward<Args>(args)...);
	}
	
private:
	std::shared_ptr<boost::python::object> object_;
	
};

BOOST_PYTHON_MODULE(PyOpenfaceVideo) {
	
    using namespace boost::python;
	
	class_<OpenfaceV3>("OpenfaceV3")
        .def_readonly("x", &OpenfaceV3::x)
        .def_readonly("y", &OpenfaceV3::y)
        .def_readonly("z", &OpenfaceV3::z);
	
    class_<OpenfaceV3Vector>("OpenfaceV3Vector")
        .def(vector_indexing_suite<OpenfaceV3Vector>());
	
	class_<OpenfaceFrame>("OpenfaceFrame")
        .def_readonly("ID", &OpenfaceFrame::fID)
        .def_readonly("gaze_0", &OpenfaceFrame::gaze_0)
        .def_readonly("gaze_1", &OpenfaceFrame::gaze_1)
        .def_readonly("head_rot", &OpenfaceFrame::head_rot)
        .def_readonly("head_pos", &OpenfaceFrame::head_pos)
        .def_readonly("landmarks", &OpenfaceFrame::landmarks);
	
	class_<OpenfaceVideo>("OpenfaceVideo")
    	.add_property("device", &OpenfaceVideo::get_device, &OpenfaceVideo::set_device)
    	.add_property("landmark_model", &OpenfaceVideo::get_landmark_model, &OpenfaceVideo::set_landmark_model)
    	.add_property("HAAR", &OpenfaceVideo::get_HAAR, &OpenfaceVideo::set_HAAR)
    	.add_property("MTCNN", &OpenfaceVideo::get_MTCNN, &OpenfaceVideo::set_MTCNN)
    	.def("start", &OpenfaceVideo::start)
    	.def("stop", &OpenfaceVideo::stop)
    	.def("is_running", &OpenfaceVideo::is_running)
    	.def("has_frame", &OpenfaceVideo::has_frame)
    	.def_readonly("frame", &OpenfaceVideo::public_frame);
}

int main(int argc, char **argv) {

	OpenfaceVideo* ov = new OpenfaceVideo();
	ov->load_arguments( argc, argv );
	ov->start();
	while( ov->is_running() ) {
		usleep( 100 );
	}
	delete ov;
	return 0;
	
}

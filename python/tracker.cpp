// the python version for FaceTrackingVidMulti.cpp : Defines the entry point for the multiple face tracking console application.
#include "tracker.hpp"

using namespace std;

vector<string> Tracker::get_arguments(int argc, char **argv)
{

	vector<string> arguments;
	for(int i = 0; i < argc; ++i)
	{
		arguments.push_back(string(argv[i]));
	}
	return arguments;
}

void Tracker::NonOverlapingDetections(const vector<LandmarkDetector::CLNF>& clnf_models, vector<cv::Rect_<double> >& face_detections)
{

	// Go over the model and eliminate detections that are not informative (there already is a tracker there)
	for(size_t model = 0; model < clnf_models.size(); ++model)
	{
		// See if the detections intersect
		cv::Rect_<double> model_rect = clnf_models[model].GetBoundingBox();
		
		for(int detection = face_detections.size()-1; detection >=0; --detection)
		{
			double intersection_area = (model_rect & face_detections[detection]).area();
			double union_area = model_rect.area() + face_detections[detection].area() - 2 * intersection_area;

			// If the model is already tracking what we're detecting ignore the detection, this is determined by amount of overlap
			if( intersection_area/union_area > 0.5)
			{
				face_detections.erase(face_detections.begin() + detection);
			}
		}
	}

}


Tracker::Tracker(vector<string> &args) : frame_count(0),
										fpsSt("FPS: "),
										done(false),
										f_n(-1),
										device(0),
										fx(600),
										fy(600),
										cx(0),
										cy(0),
										num_faces_max(4),
										all_models_active(true),
										active_models_st("Active Models: "),
										num_active_models(0)
{

	LandmarkDetector::FaceModelParameters det_params(args);
	det_params.use_face_template = true;
	// This is so that the model would not try re-initialising itself
	det_params.reinit_video_every = -1;
	det_params.curr_face_detector = LandmarkDetector::FaceModelParameters::HOG_SVM_DETECTOR;
	det_parameters.push_back(det_params);

	LandmarkDetector::get_video_input_output_params(files, depth_directories, dummy_out, tracked_videos_output, u, output_codec, arguments);
	use_depth = !depth_directories.empty();
	// Get camera parameters
	LandmarkDetector::get_camera_params(device, fx, fy, cx, cy, arguments);

	LandmarkDetector::CLNF clnf_model(det_parameters[0].model_location);
	clnf_model.face_detector_HAAR.load(det_parameters[0].face_detector_location);
	clnf_model.face_detector_location = det_parameters[0].face_detector_location;
	clnf_models.reserve(num_faces_max);
	clnf_models.push_back(clnf_model);

	active_models.push_back(false);

	for (int i = 1; i < num_faces_max; ++i)
	{
		clnf_models.push_back(clnf_model);
		active_models.push_back(false);
		det_parameters.push_back(det_params);
	}

	// Get depth image
	if(use_depth)
	{
		char* dst = new char[100];
		std::stringstream sstream;

		sstream << depth_directories[f_n] << "\\depth%05d.png";
		sprintf(dst, sstream.str().c_str(), frame_count + 1);
		// Reading in 16-bit png image representing depth
		depth_image_16_bit = cv::imread(string(dst), -1);

		// Convert to a floating point depth image
		if(!depth_image_16_bit.empty())
		{
			depth_image_16_bit.convertTo(depth_image, CV_32F);
		}
		else
		{
			WARN_STREAM( "Can't find depth image" );
		}
	}
	
	if(cx == 0 || cy == 0)
	{
		cx_undefined = true;
	}		
	// If optical centers are not defined just use center of image
	if(cx_undefined)
	{
		cx = captured_image.cols / 2.0f;
		cy = captured_image.rows / 2.0f;
	}

}

bool Tracker::tracking(cv::Mat &video_capture,
					   vector<cv::Rect_<double>> &face_rects,
					   vector<cv::Mat_<double>> &face_landmarks)
{

	// We might specify multiple video files as arguments
	captured_image = video_capture;
	disp_image = captured_image.clone();
	if(files.size() > 0)
	{
		f_n++;			
	    current_file = files[f_n];
	}

	//start handle
	if(!tracked_videos_output.empty())
	{
		try
		{
			writerFace = cv::VideoWriter(tracked_videos_output[f_n], CV_FOURCC(output_codec[0],output_codec[1],output_codec[2],output_codec[3]), 30, captured_image.size(), true);
		}
		catch(cv::Exception e)
		{
			WARN_STREAM( "Could not open VideoWriter, OUTPUT FILE WILL NOT BE WRITTEN. Currently using codec " << output_codec << ", try using an other one (-oc option)");
		}
	}
	
	// For measuring the timings
	int64 t1,t0 = cv::getTickCount();
	double fps = 10;

	INFO_STREAM( "Starting tracking");

	if(captured_image.channels() == 3)
	{
		cv::cvtColor(captured_image, grayscale_image, CV_BGR2GRAY);				
	}
	else
	{
		grayscale_image = captured_image.clone();				
	}
	
	for(unsigned int model = 0; model < clnf_models.size(); ++model)
	{
		if(!active_models[model])
		{
			all_models_active = false;
		}
	}
				
	// Get the detections (every 8th frame and when there are free models available for tracking)
	if(frame_count % 8 == 0 && !all_models_active)
	{				
		if(det_parameters[0].curr_face_detector == LandmarkDetector::FaceModelParameters::HOG_SVM_DETECTOR)
		{
			vector<double> confidences;
			LandmarkDetector::DetectFacesHOG(face_detections, grayscale_image, clnf_models[0].face_detector_HOG, confidences);
		}
		else
		{
			LandmarkDetector::DetectFaces(face_detections, grayscale_image, clnf_models[0].face_detector_HAAR);
		}

	}

	// Keep only non overlapping detections (also convert to a concurrent vector
	NonOverlapingDetections(clnf_models, face_detections);

	vector<tbb::atomic<bool> > face_detections_used(face_detections.size());

	// Go through every model and update the tracking
	tbb::parallel_for(0, (int)clnf_models.size(), [&](int model){
	//for(unsigned int model = 0; model < clnf_models.size(); ++model)
	//{

		bool detection_success = false;

		// If the current model has failed more than 4 times in a row, remove it
		if(clnf_models[model].failures_in_a_row > 4)
		{				
			active_models[model] = false;
			clnf_models[model].Reset();

		}

		// If the model is inactive reactivate it with new detections
		if(!active_models[model])
		{
			
			for(size_t detection_ind = 0; detection_ind < face_detections.size(); ++detection_ind)
			{
				// if it was not taken by another tracker take it (if it is false swap it to true and enter detection, this makes it parallel safe)
				if(face_detections_used[detection_ind].compare_and_swap(true, false) == false)
				{
			
					// Reinitialise the model
					clnf_models[model].Reset();

					// This ensures that a wider window is used for the initial landmark localisation
					clnf_models[model].detection_success = false;
					detection_success = LandmarkDetector::DetectLandmarksInVideo(grayscale_image, depth_image, face_detections[detection_ind], clnf_models[model], det_parameters[model]);
											
					// This activates the model
					active_models[model] = true;

					// break out of the loop as the tracker has been reinitialised
					break;
				}

			}
		}
		else
		{
			// The actual facial landmark detection / tracking
			detection_success = LandmarkDetector::DetectLandmarksInVideo(grayscale_image, depth_image, clnf_models[model], det_parameters[model]);
		}
	});
						
	// Go through every model and visualise the results
	for(size_t model = 0; model < clnf_models.size(); ++model)
	{
		// Visualising the results
		// Drawing the facial landmarks on the face and the bounding box around it if tracking is successful and initialised
		double detection_certainty = clnf_models[model].detection_certainty;

		double visualisation_boundary = -0.1;
	
		// Only draw if the reliability is reasonable, the value is slightly ad-hoc
		if(detection_certainty < visualisation_boundary)
		{
			LandmarkDetector::Draw(disp_image, clnf_models[model]);

			if(detection_certainty > 1)
				detection_certainty = 1;
			if(detection_certainty < -1)
				detection_certainty = -1;

			detection_certainty = (detection_certainty + 1)/(visualisation_boundary +1);

			// A rough heuristic for box around the face width
			int thickness = (int)std::ceil(2.0* ((double)captured_image.cols) / 640.0);
			
			// Work out the pose of the head from the tracked model
			cv::Vec6d pose_estimate = LandmarkDetector::GetCorrectedPoseWorld(clnf_models[model], fx, fy, cx, cy);
			
			// Draw it in reddish if uncertain, blueish if certain
			LandmarkDetector::DrawBox(disp_image, pose_estimate, cv::Scalar((1-detection_certainty)*255.0,0, detection_certainty*255), thickness, fx, fy, cx, cy);

			cv::Mat_<double> landmarks_2d = clnf_models[model].detected_landmarks;
			landmarks_2d = landmarks_2d.reshape(1, 2);
			cv::Rect_<double> rect = clnf_models[model].GetBoundingBox();
			face_rects.push_back(rect);
			face_landmarks.push_back(landmarks_2d);
		}
	}

	// Work out the framerate
	if(frame_count % 10 == 0)
	{      
		t1 = cv::getTickCount();
		fps = 10.0 / (double(t1-t0)/cv::getTickFrequency()); 
		t0 = t1;
	}
	
	// Write out the framerate on the image before displaying it
	sprintf(fpsC, "%d", (int)fps);
	fpsSt += fpsC;
	cv::putText(disp_image, fpsSt, cv::Point(10,20), CV_FONT_HERSHEY_SIMPLEX, 0.5, CV_RGB(255,0,0), 1, CV_AA);
	
	num_active_models = 0;

	for( size_t active_model = 0; active_model < active_models.size(); active_model++)
	{
		if(active_models[active_model])
		{
			num_active_models++;
		}
	}

	sprintf(active_m_C, "%d", num_active_models);
	active_models_st += active_m_C;
	cv::putText(disp_image, active_models_st, cv::Point(10,60), CV_FONT_HERSHEY_SIMPLEX, 0.5, CV_RGB(255,0,0), 1, CV_AA);
	
	if(!det_parameters[0].quiet_mode)
	{
		cv::namedWindow("tracking_result",1);
		cv::imshow("tracking_result", disp_image);

		if(!depth_image.empty())
		{
			// Division needed for visualisation purposes
			imshow("depth", depth_image/2000.0);
		}
	}
	//char character_press = cv::waitKey(1000);

	// output the tracked video
	if(!tracked_videos_output.empty())
	{		
		writerFace << disp_image;
	}
	// Update the frame count
	frame_count++;

	if(face_rects.size()>0 && face_landmarks.size()> 0)
		return true;
	else
		return false;

	//end handle
}


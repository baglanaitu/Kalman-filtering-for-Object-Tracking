/* Applied Video Analysis of Sequences (AVSA)
 *
 *	LAB2: Blob detection & classification
 *	Lab2.0: Sample Opencv project
 *
 *
 * Authors: José M. Martínez (josem.martinez@uam.es), Paula Moral (paula.moral@uam.es), Juan C. San Miguel (juancarlos.sanmiguel@uam.es)
 */

//system libraries C/C++
#include <stdio.h>
#include <iostream>
#include <sstream>
#include <utility>
#include <vector>
#include <stdlib.h>

//opencv libraries
#include <opencv2/opencv.hpp>
#include <opencv2/video/background_segm.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/core/mat.hpp"
#include <opencv2/video/tracking.hpp>

//Header ShowManyImages
#include "ShowManyImages.hpp"

//include for blob-related functions
#include "blobs.hpp"

//namespaces
using namespace cv; //avoid using 'cv' to declare OpenCV functions and variables (cv::Mat or Mat)
using namespace std;

#define MIN_WIDTH 10
#define MIN_HEIGHT 10


//main function
int main(int argc, char ** argv)
{
	Mat frame; // current Frame
	Mat fgmask; // foreground mask
	std::vector<cvBlob> bloblist; // list for blobs
	std::vector<cvBlob> filtered_bloblist; // list for blobs

	// STATIONARY BLOBS
	Mat fgmask_history; // STATIONARY foreground mask
	Mat sfgmask; // STATIONARY foreground mask
	std::vector<cvBlob> sbloblist; // list for STATIONARY blobs
	std::vector<cvBlob> sbloblistFiltered; // list for STATIONARY blobs


	double t, acum_t; //variables for execution time
		int t_freq = getTickFrequency();

		//Paths for the dataset
		string dataset_path = "/home/avsa/Documents/AVSA_Lab3_datasets";
		string dataset_cat[1] = {"dataset_lab3"};
		string baseline_seq[1] = {"lab3.1/singleball"};
		string image_path = ".mp4"; //path to images - this format allows to read consecutive images with filename inXXXXXX.jpq (six digits) starting with 000001


		//Paths for the results
		string project_root_path = "/home/avsa/Documents"; //SET THIS DIRECTORY according to your project
		string project_name = "Lab3.1AVSA2020"; //SET THIS DIRECTORY according to your project
		string results_path = project_root_path+"/"+"AVSA_Lab3_results";


		int NumCat = sizeof(dataset_cat)/sizeof(dataset_cat[0]); //number of categories (have faith ... it works! ;) ... each string size is 32 -at leat for the current values-)

		//Loop for all categories
		//Loop for all categories
		for (int c=0; c<NumCat; c++ )
		{

			int NumSeq = sizeof(baseline_seq)/sizeof(baseline_seq[0]);  //number of sequences per category ((have faith ... it works! ;) ... each string size is 32 -at leat for the current values-)

			//Loop for all sequence of each category
			for (int s=0; s<NumSeq; s++ )
			{
			VideoCapture cap;//reader to grab videoframes

			//Concatenate path of images
			string inputvideo = dataset_path + "/" + dataset_cat[c] + "/" + baseline_seq[s] + image_path;
			cout << "Accessing sequence at " << inputvideo << endl;

			//open the video file to check if it exists
			cap.open(inputvideo);
			if (!cap.isOpened()) {
				cout << "Could not open video file " << inputvideo << endl;
			return -1;
			}


			// create directroy to store the results
			string makedir_cmd = "mkdir -p "+results_path + "/" + dataset_cat[c] + "/" + baseline_seq[s];
			system(makedir_cmd.c_str());

			//PARAMETERS:
			double mog_learning_rate = 0.001;
			int opening_kernel_size = 3;
			bool write_result=true; //depending on if we want to save the results or not
			bool imageByImage = false; //depending on if we want to see it as a video or image by image

			//MOG2 parameters
			double varThreshold = 16;
			int history = 50;
			Ptr<BackgroundSubtractor> pMOG2 = cv::createBackgroundSubtractorMOG2(history, varThreshold, true);

			/*
			 * Select Kalman filter and corresponding state_size:
			 * - CVKF -> state_size = 4
			 * - CAKF -> state_size = 6
			 * */

			// KalmanFilter KF = CVKF();
			KalmanFilter KF = CAKF();
			// KalmanFilter KF = cvCreateKalman();
			int state_size = 6;

			std::vector<Point2f> predicted_points; //the container with the trajectory
			std::vector<Point2f> measured_points; //the container with the measurements
			cv::Mat _background; // the background, the first frame (used only for visualization)
			bool has_started = false;

			//main loop
			Mat img; // current Frame

			int it=1;
			acum_t=0;
			for (;;) {
				//get frame
				cap >> img;
				//check if we achieved the end of the file (e.g. img.data is empty)
				if (!img.data)
					break;

				//Time measurement
				t = (double)getTickCount();

				//apply algs
				img.copyTo(frame);
				// Compute fgmask
				double learningrate = mog_learning_rate; //default value (as starting point)
				// The value between 0 and 1 that indicates how fast the background model is
				// learnt. Negative parameter (default -1) value makes the algorithm to use some automatically chosen learning
				// rate. 0 means that the background model is not updated at all, 1 means that the background model
				// is completely reinitialized from the last frame.
				pMOG2->apply(frame, fgmask, learningrate);
				Mat opening_kernel = getStructuringElement(MORPH_RECT, Size(opening_kernel_size,opening_kernel_size));
				morphologyEx(fgmask, fgmask, MORPH_OPEN, opening_kernel);

				int connectivity = 8; // 4 or 8 for the flood fill algorithm

				// Extract the blobs, object candidates for tracking
				extractBlobs(fgmask, bloblist, connectivity);

				Point2f measurement_pair = removeSmallBlobs(bloblist, filtered_bloblist, MIN_WIDTH, MIN_HEIGHT);
				Point2f currentPoint(-1,-1);

				bool hasMeasurement = false;
				runKF(measurement_pair, KF, measured_points, predicted_points, has_started, hasMeasurement, currentPoint, state_size);

				// STATIONARY BLOBS
				if (it==1)
					{
					sfgmask = Mat::zeros(Size(fgmask.cols, fgmask.rows), CV_8UC1);
					fgmask_history = Mat::zeros(Size(fgmask.cols, fgmask.rows), CV_32FC1);
					img.copyTo(_background);
					}


				//Time measurement
				t = (double)getTickCount() - t;
//		        if (_CONSOLE_DEBUG) cout << "proc. time = " << 1000*t/t_freq << " milliseconds."<< endl;
				acum_t=+t;

				//SHOW RESULTS
				//get the frame number and write it on the current frame

				stringstream it_ss;
				it_ss << setw(6) << setfill('0') << it;

				string title= project_name + " " +dataset_cat[c] + "/" + baseline_seq[s];// + it_ss.str();

				ShowManyImages(title, 6, frame, fgmask, sfgmask,
						paintBlobImage(frame,currentPoint, hasMeasurement, filtered_bloblist), paintBlobImage(frame,currentPoint, hasMeasurement, filtered_bloblist), sfgmask);

				// Trajectory image
				string outFile = "/home/avsa/Documents/AVSA_Lab3_results/" + dataset_cat[c] + "/" + baseline_seq[s] + "/out"+ it_ss.str() +".png";

				if(write_result) {
					write_result=imwrite(outFile, paintBlobImage(frame,currentPoint, hasMeasurement, filtered_bloblist));
				}
				if(imageByImage){
					waitKey(0);
				}


				//exit if ESC key is pressed
				if(waitKey(30) == 27) break;
				it++;
			} //main loop

	cout << it-1 << "frames processed in " << 1000*acum_t/t_freq << " milliseconds."<< endl;
	string nameOfTrajectory = "/home/avsa/Documents/AVSA_Lab3_results/" + dataset_cat[c] + "/" + baseline_seq[s] +".png";
	paintTrajectory(_background, predicted_points, measured_points, nameOfTrajectory);


	//release all resources

	cap.release();
	destroyAllWindows();
	waitKey(0); // (should stop till any key is pressed .. doesn't!!!!!)
}
}
return 0;
}




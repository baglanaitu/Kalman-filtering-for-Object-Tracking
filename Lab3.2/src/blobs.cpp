#include "blobs.hpp"
#include <cmath>
#include <algorithm>
#include <utility>
#include <math.h>


 Mat paintBlobImage(cv::Mat frame, Point2f predicted_point, bool labelled, std::vector<cvBlob> bloblist)
{
	cv::Mat Image;
	int height = 50;
	int width = 50;

	frame.copyTo(Image);
	if(bloblist.size() > 0) {
		height = floor(bloblist[0].h/2);
		width = floor(bloblist[0].w/2);
	}

	if(predicted_point.x != -1) {
		Scalar color(255, 255, 255);

		// show the tracking with a red ellipse
		ellipse(Image, predicted_point, Size(width, height), 0, 0, 360, Scalar( 0, 0, 255), 4, LINE_8);
		if (labelled) {
			putText(Image, "Corrected", predicted_point, FONT_HERSHEY_SIMPLEX, 3, color, 2.0);
		}
		else {
			putText(Image, "Predicted", predicted_point, FONT_HERSHEY_SIMPLEX, 3, color, 2.0);
		}
	}


	return Image;
}

void paintTrajectory(cv::Mat _background, std::vector<Point2f> pointTrajectory, std::vector<Point2f> measuredPoints, std::string output_path) {

	if(measuredPoints.size() >1) {

		cv::drawMarker(_background, measuredPoints[0], Scalar( 255, 0, 0),MARKER_CROSS , 6, 4, LINE_8);

		for (int i = 1; i < measuredPoints.size(); i++) {

			Point2f pt = measuredPoints[i];
			cv::line(_background, measuredPoints[i-1], measuredPoints[i], Scalar( 255, 0, 0), 2, LINE_8);
			cv::drawMarker(_background, measuredPoints[i], Scalar( 255, 0, 0), MARKER_CROSS, 6, 4, LINE_8);
		}
	}

	if (pointTrajectory.size() > 1) {

		circle(_background, pointTrajectory[0], 4, Scalar( 0, 0, 255), 1, LINE_8);

		for (int i = 1; i < pointTrajectory.size(); i++){
			Point2f pt = pointTrajectory[i];
			cv::line(_background, pointTrajectory[i-1], pointTrajectory[i], Scalar( 0, 0, 255), 2, LINE_8);
			circle(_background, pt, 4, Scalar( 0, 0, 255), 2, LINE_8);

		}
	}

	namedWindow("Tracking results", 1 );
	putText(_background, "Measured x_k", Point(5,25), FONT_HERSHEY_PLAIN, 2.0, Scalar( 255, 0, 0), 2.0);
	putText(_background, "Estimated x_k", Point(5,50), FONT_HERSHEY_PLAIN, 2.0, Scalar( 0, 0, 255), 2.0);
	imshow("Tracking results", _background);

	imwrite(output_path, _background);
}

// RECURSIVE FLOODFILL

//x,y: seed pixel (or current)
//aux: the image we want to iterate on
//maxX,Y minX,Y: stores the smallest rectangle contains our blob
//connectivity: what kind of neighbourhood to examine
//area: stores the amount of pixels, that our blob covers.
void recursive_floodfill(int x, int y, Mat& frame, int& maxX, int& maxY, int& minX, int& minY, int connectivity, int &area) {

	 if(x >= 0 && y >= 0 && x < frame.rows && y < frame.cols && frame.at<int>(x,y) == 255){
		 if(x >= maxX)
			 maxX = x;
		 if(y >= maxY)
			 maxY = y;

		 if(x <= minX)
			 minX = x;

		 if(y <= minY)
			 minY = y;

		 area++;
		 frame.at<int>(x, y) = 0;

		 if(connectivity == 8) recursive_floodfill(x-1,y-1,frame, maxX, maxY, minX, minY, connectivity, area);

		 recursive_floodfill(x,y-1,frame, maxX, maxY, minX, minY, connectivity, area);

		 if(connectivity == 8) recursive_floodfill(x+1,y-1,frame, maxX, maxY, minX, minY, connectivity, area);

		 recursive_floodfill(x-1,y,frame, maxX, maxY, minX, minY, connectivity, area);
		 recursive_floodfill(x+1,y,frame, maxX, maxY, minX, minY, connectivity, area);

		 if(connectivity == 8) recursive_floodfill(x-1,y+1,frame, maxX, maxY, minX, minY, connectivity, area);
		 recursive_floodfill(x,y,frame, maxX, maxY, minX, minY, connectivity, area);
		 if(connectivity == 8) recursive_floodfill(x+1,y+1,frame, maxX, maxY, minX, minY, connectivity, area);
	 }
 }


// Extract the blobs, object candidates for tracking
int extractBlobs(cv::Mat fgmask, std::vector<cvBlob> &bloblist, int connectivity) {

	Mat aux; // image to be updated each time a blob is detected (blob cleared)
	fgmask.convertTo(aux,CV_32SC1);

	cv::Mat img = cv::Mat::zeros(aux.cols,aux.rows,CV_8UC1);

	bloblist.clear();

	//Connected component analysis
	int counter = 0;
	for(int i = 0; i < fgmask.rows; i++){
		for(int j = 0; j < fgmask.cols; j++){
			if(aux.at<int>(i,j) == 255){
				cv::Point seed(j,i);
				cv::Rect rect;
				//cv::recursive_floodfill(aux, seed, 0, &rect, cv::Scalar(1), cv::Scalar(1), connectivity);
				//cvBlob blob=initBlob(counter, rect.x, rect.y, rect.width, rect.height);
				int maxX =i;
				int minX = i;
				int maxY = j;
				int minY = j;
				int area = 0;

				recursive_floodfill(i, j, aux, maxX, maxY, minX, minY, connectivity, area);
				counter++;
				cvBlob blob=initBlob(counter, minY, minX, maxY-minY, maxX-minX, area);
				//collect detected blobs:
				bloblist.push_back(blob);
			}
		}
	}
	return 1;
}



// We get the biggest blob
Point2f removeSmallBlobs(std::vector<cvBlob> bloblist_in, std::vector<cvBlob> &bloblist_out, int min_width, int min_height)
{
	bloblist_out.clear();
	std::pair<int,int> center(NULL,NULL);
	Point2f pt(NULL, NULL);
	//If there are no detections at all, dont start working
	if (bloblist_in.size() == 0){

		return pt;
	}
	int indexWithBiggestArea = -1;
	int max_area = 0;
	//iterates on the blobs, and finds the biggest one, which is also bigger than the min_width and min_height
	for(int i = 0; i < bloblist_in.size(); i++)
	{
		//int current_area = bloblist_in[i].h * bloblist_in[i].w;
		int current_area = bloblist_in[i].area;
		if ( current_area >  max_area && bloblist_in[i].h > min_height && bloblist_in[i].w > min_width){
			indexWithBiggestArea = i;
			max_area = current_area;
		}

	}
	//if there are none above the threshold
	if(indexWithBiggestArea == -1){
		return pt;
	}
	cvBlob finalBlob = bloblist_in[indexWithBiggestArea];
	//center.first = floor(finalBlob.x+finalBlob.w/2);
	//center.second = floor(finalBlob.y+finalBlob.h/2);

	//put it in a comfortable format
	pt.x = finalBlob.x+finalBlob.w/2;
	pt.y = finalBlob.y+finalBlob.h/2;
	bloblist_out.push_back(finalBlob);
	return pt;
}

// Constant Velocity Kalman Filter (CVKF)
KalmanFilter CVKF() {

	KalmanFilter KF(4, 2, 0);

	KF.transitionMatrix = (Mat_<float>(4,4) << // A, 4x4
			1, 1, 0, 0,
			0, 1, 0, 0,
			0, 0, 1, 1,
			0, 0, 0, 1);
	KF.measurementMatrix = (Mat_<float>(2,4) <<
			1, 0, 0, 0,
			0, 0, 1, 0); //H 2x4

	KF.processNoiseCov = (Mat_<float>(4,4) << // Q, 4x4
			25, 0, 0, 0,
			0, 10, 0, 0,
			0, 0, 25, 0,
			0, 0, 0, 10);

	setIdentity(KF.measurementNoiseCov, Scalar::all(25)); //R, 2x2

	return KF;
}

// Cosnstant Acceleration Kalman Filter (CAKF)
KalmanFilter CAKF(){
	KalmanFilter KF(6, 2, 0);
	KF.transitionMatrix = (Mat_<float>(6,6) << // A, 6x6
			1, 1, 0.5, 0, 0, 0,
			0, 1, 1, 0, 0, 0,
			0, 0, 1, 0, 0, 0,
			0, 0, 0, 1, 1, 0.5,
			0, 0, 0, 0, 1, 1,
			0, 0, 0, 0, 0, 1);

	KF.measurementMatrix = (Mat_<float>(2,6) << //H 2x6
			1, 0, 0, 0, 0, 0,
			0, 0, 0, 1, 0, 0);


	KF.processNoiseCov = (Mat_<float>(6,6) << // Q, 6x6
				25, 0, 0, 0, 0 , 0,
				0, 10, 0, 0, 0, 0,
				0, 0, 1, 0, 0, 0,
				0, 0, 0, 25, 0, 0,
				0, 0, 0, 0, 10, 0,
				0, 0, 0, 0, 0, 1);

	setIdentity(KF.measurementNoiseCov, Scalar::all(25)); //R, 2x2

	return KF;
}


void runKF(Point2f measurement_pair, KalmanFilter &KF, std::vector<Point2f> &measured_points, std::vector<Point2f> &predicted_points, bool &has_started, bool &has_Measurement, Point2f &currentPoint, int state_size){

	Mat state(state_size, 1, CV_32F);

	//INIT THE KALMAN FILTER
	if(measurement_pair.x != NULL && !has_started) {
			//it is initialited with the current measurement as center, and random speed
		if(state_size == 4) {
			KF.statePost = (Mat_<float>(4,1) << measurement_pair.x, rand() % 5 -2, measurement_pair.y, rand() % 5 -2);
		}

		if(state_size == 6) {
			currentPoint.x = state.at<float>(0);

			currentPoint.y = state.at<float>(2);KF.statePost = (Mat_<float>(state_size,1) << measurement_pair.x,
				rand() % 5 -2,
				rand() % 3 -1,
				measurement_pair.y,
				rand() % 5 -2,
				rand() % 3 -1);
		}
		setIdentity(KF.errorCovPre, Scalar::all(1e5)); // P_0, 4x4 or 6x6 depending on the state_size

		has_started = true;
	}else if(has_started){
		// if the tracking is working we predict the next state based on the model
		Mat prediction = KF.predict();

		if(measurement_pair.x != NULL){ // We had measurement, thus we CORRECT !!
			Mat measurement = Mat::zeros(2, 1, CV_32F);

			measured_points.push_back(measurement_pair);
			measurement.at<float>(0) = measurement_pair.x;
			measurement.at<float>(1) = measurement_pair.y;
			std::cout << "Measurement\n";
			std::cout << measurement_pair.x << " " << measurement_pair.y << std::endl;
			state = KF.correct(measurement);

			has_Measurement = true;
		}else { // NO MEASUREMENT, STATE WILL BE THE PREDICTION UNTIL HAS_MEASUREMENT = TRUE
					// if there were no measurement, the state will be only the prediction
			state = prediction;
		}
		std::cout << "State:" << std::endl;
		for(int i = 0; i < 2; i++){
			std::cout << state.at<float>(i) << " ";
		}
		std::cout << "\n*-*-*-*-*-" << std::endl;

		if (state_size == 4) { // CV KF [x,x',y,y']
			currentPoint.x = state.at<float>(0);
			currentPoint.y = state.at<float>(2);

		}
		else { // CA KF [x,x',x'',y,y',y'']
			currentPoint.x = state.at<float>(0);
			currentPoint.y = state.at<float>(3);
		}
		// we save the predictions
		predicted_points.push_back(currentPoint);
	}
}








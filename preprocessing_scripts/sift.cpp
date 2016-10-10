#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <iostream>
#include <fstream>

using namespace cv;
using namespace std;

int WINDOW_SIZE = 32;


int save_sample(int x, int y, int sample_number) {
	
}

int main(int argc, const char* argv[])
{
    const cv::Mat input = cv::imread(argv[1], 0); //Load as grayscale

    SiftFeatureDetector detector(151);
    vector<KeyPoint> keypoints;
    detector.detect(input, keypoints);

    ofstream fout;
    fout.open(argv[2]);

    for(int i=0;i<keypoints.size();i++){
    	int x = keypoints[i].pt.x;
    	int y = keypoints[i].pt.y;
    	
        fout << x << " " << y << endl;
    	//save_sample(x,y,i);
    }

    fout.close();

    //FileStorage fs(argv[2],FileStorage::WRITE);
    //write(fs,"keypointss",keypoints);
    //fs.release();
    // Add results to image and save.
    //cv::Mat output;
    //cv::drawKeypoints(input, keypoints, output);
    //cv::imwrite("sift_result.jpg", output);

    return 0;
}

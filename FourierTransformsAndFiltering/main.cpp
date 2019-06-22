#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

void perfromDFT(Mat &src, Mat &dst);
void computeMagnitude(Mat &src, Mat &dst);
void rearrangeQuadFourierImg(Mat &src);



void perfromDFT(Mat &src, Mat &dst) {
	Mat padded;                            //expand input image to optimal size
	int m = getOptimalDFTSize(src.rows);
	int n = getOptimalDFTSize(src.cols); // on the border add zero values
	copyMakeBorder(src, padded, 0, m - src.rows, 0, n - src.rows, BORDER_CONSTANT, Scalar::all(0));
	Mat planes[] = { Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F) };
	merge(planes, 2, dst);         // Add to the expanded another plane with zeros
	dft(dst, dst);            // this way the result may fit in the source matrix

	//releasing allocated memory
	padded.release();
	planes->release();
}




void computeMagnitude(Mat &src, Mat &dst){
	// compute the magnitude and switch to logarithmic scale
	// => log(1 + sqrt(Re(DFT(I))^2 + Im(DFT(I))^2))
	Mat planes[] = { Mat::zeros(src.size(), CV_32F), Mat::zeros(src.size(), CV_32F)	};
	split(src, planes);                   // planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
	magnitude(planes[0], planes[1], planes[0]);// planes[0] = magnitude
	dst = planes[0];		//magnitude image

	dst += Scalar::all(1);                    // switch to logarithmic scale
	log(dst, dst);

	rearrangeQuadFourierImg(dst);

	normalize(dst, dst, 0, 1, NORM_MINMAX); // Transform the matrix with float values into a
										// viewable image form (float between values 0 and 1).

	planes->release();
}



void rearrangeQuadFourierImg(Mat &src) {
	// crop the spectrum, if it has an odd number of rows or columns
	src = src(Rect(0, 0, src.cols & -2, src.rows & -2));

	// rearrange the quadrants of Fourier image  so that the origin is at the image center
	int cx = src.cols / 2;
	int cy = src.rows / 2;

	Mat q0(src, Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
	Mat q1(src, Rect(cx, 0, cx, cy));  // Top-Right
	Mat q2(src, Rect(0, cy, cx, cy));  // Bottom-Left
	Mat q3(src, Rect(cx, cy, cx, cy)); // Bottom-Right

	Mat tmp;                           // swap quadrants (Top-Left with Bottom-Right)
	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);

	q1.copyTo(tmp);                    // swap quadrant (Top-Right with Bottom-Left)
	q2.copyTo(q1);
	tmp.copyTo(q2);

	//Relesing memory
	q0.release();
	q1.release();
	q2.release();
	q3.release();
	tmp.release();
}



int main() {

	//Reading images
	//string imgDir = "C:\\Users\\mimtiaz\\visualStudio17Projects\\getMeHired\\computerVision\\standard-test-images-for-Image-Processing\\standard_test_images\\lena.bmp";
	string imgDir = "C:\\Users\\mimtiaz\\visualStudio17Projects\\getMeHired\\computerVision\\letters\\B2.png";
	Mat img = imread(imgDir, IMREAD_COLOR);

	

	//checking for errors
	if (img.empty()) {
		printf("Error Reading images\n");
		return -1;
	}

	//converting to gray
	cvtColor(img, img, COLOR_BGR2GRAY);

	Mat complexImg, magImg;
	//Perform DFT
	perfromDFT(img, complexImg);
	//Compute magnitude and convert into log
	computeMagnitude(complexImg, magImg);


	imshow("Input Image", img);    // Show the result
	imshow("spectrum magnitude", magImg);
	waitKey(0);

	return 0;
}
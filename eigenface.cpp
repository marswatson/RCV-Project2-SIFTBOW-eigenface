#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/legacy/legacy.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/contrib/contrib.hpp>
#include <vector>
#include <iomanip>


using namespace cv;
using namespace std;


int main() {
	// use to input training data
	char * filename = new char[100];
	Mat input;
	vector<Mat> imagesTrain;
	vector<int> labelsTrain;

	//input train data
	for (int j = 1; j < 6; j++)
		for (int i = 1; i <= 20; i += 2){
		sprintf(filename, "face\\yaleface (%i).pgm", i + (j - 1) * 20);
		input = imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
		imagesTrain.push_back(input);
		labelsTrain.push_back(j);
		}
	//imshow("a",imagesTrain[0]);
	//create eigenface detector
	Ptr<FaceRecognizer> model = createEigenFaceRecognizer();
	model->train(imagesTrain, labelsTrain);

	// test other face from database
	vector<Mat> imagesTest;
	vector<int> labelsTest;
	vector<double> confidenceTest;

	//input test data
	for (int j = 1; j < 6; j++)
		for (int i = 2; i <= 20; i += 2){
		sprintf(filename, "face\\yaleface (%i).pgm", i + (j - 1) * 20);
		input = imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
		int predictedLabel;
		//it can caculate confidence of test
		double confidence = 0;
		model->predict(input, predictedLabel,confidence);
		imagesTest.push_back(input);
		labelsTest.push_back(predictedLabel);
		confidenceTest.push_back(confidence);
		}

	//caculate the error rate of eigenface recognition
	double errorRate;
	int count = 0;
	for (int i = 0; i < 50; i++)
	{
		if (labelsTrain[i] - labelsTest[i] != 0)
			count++;
	}
	errorRate = double(count) / imagesTest.size();
	cout << "Total error rate is: " << errorRate << endl;

	//print confusion matrix
	float Probability;
	cout << " " << setw(8) << "1" << setw(8) << "2" << setw(8) << "3" << setw(8) << "4" << setw(8) << "5" << endl;
	for (int i = 1; i <= 5; i++){
		cout << i;
		for (int j = 1; j <= 5; j++){
			count = 0;
			for (int k = 1; k <= 10; k++){
				if (labelsTest[(i - 1) * 10 + k - 1] == j)
					count++;
			}
			Probability = float(count) / 10.0;
			cout << setw(8) << Probability;
		}
		cout << "\n";
	}

	//get eigenface and mean face and eigenvalues from FaceRecognize
	Mat eigenvalues = model->getMat("eigenvalues");
	Mat W = model->getMat("eigenvectors");
	Mat mean = model->getMat("mean");

	int height = imagesTrain[0].rows;
	Mat norm_mean = mean.reshape(1, height);
	normalize(norm_mean, norm_mean, 0, 255, NORM_MINMAX, CV_8UC1);
	imshow("mean", norm_mean);
	// display ten eigenface
	for (int i = 0; i < min(10, W.cols); i++) {
		string msg = format("Eigenvalue #%d = %.5f", i, eigenvalues.at<double>(i));
		cout << msg << endl;
		// get #i eigenface
		Mat ev = W.col(i).clone();
		//normalize eigenvector, convert it to same size as input
		ev = ev.reshape(1, height);
		normalize(ev, ev, 0, 255, NORM_MINMAX, CV_8UC1);
		Mat grayscale = ev;

		imshow(format("eigenface_%d", i), grayscale);
	}

	// reconstruct face according, 10, 20, 30, 40 eigenvector
	for (int num_components = 10; num_components <50; num_components += 10) {

		Mat evs = Mat(W, Range::all(), Range(0, num_components));
		Mat projection = subspaceProject(evs, mean, imagesTest[0].reshape(1, 1));
		Mat reconstruction = subspaceReconstruct(evs, mean, projection);
		// normalize
		reconstruction = reconstruction.reshape(1, imagesTest[0].rows);
		normalize(reconstruction, reconstruction, 0, 255, NORM_MINMAX, CV_8UC1);

		imshow(format("eigenface_reconstruction_%d", num_components), reconstruction);
	}
	imshow("test image", imagesTest[0]);

	waitKey(0);

	return 0;
}
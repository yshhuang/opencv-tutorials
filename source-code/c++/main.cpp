#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/opencv.hpp"
#include <algorithm>

#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;
using std::cout;
using std::endl;


String local_dir = "/Volumes/develop/data/similarity/";

//int main(int argc, char *argv[]) {
//    Mat Test = (Mat_<double>(3,3) << 0,1,2, 3,4,5, 6,7,8);
//    imwrite("test.png",Test);
//}


int main() {
    std::cout << "This tutorial code needs the xfeatures2d contrib module to be run." << std::endl;
    return 0;
}

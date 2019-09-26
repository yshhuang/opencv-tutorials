//
// Created by yshhuang on 2019-08-07.
//

#include <iostream>
#include "opencv2/core.hpp"

#include <algorithm>

#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"

using namespace cv;
using namespace cv::xfeatures2d;
using std::cout;
using std::endl;


String local_dir = "/Volumes/develop/data/similarity/";

int main(int argc, char *argv[]) {
//    CommandLineParser parser( argc, argv, keys );
    Mat img2 = imread(local_dir + "838441122236728704-6", IMREAD_GRAYSCALE);
    Mat img1 = imread(local_dir + "838441122236728704-6-resize50%", IMREAD_GRAYSCALE);
    if (img1.empty() || img2.empty()) {
        cout << "Could not open or find the image!\n" << endl;
//        parser.printMessage();
        return -1;
    }
    //-- Step 1: Detect the keypoints using SURF Detector, compute the descriptors
    int minHessian = 400;
    Ptr<SURF> detector = SURF::create(minHessian);
    std::vector<KeyPoint> keypoints1, keypoints2;
    Mat descriptors1, descriptors2;
    detector->detectAndCompute(img1, noArray(), keypoints1, descriptors1);
    detector->detectAndCompute(img2, noArray(), keypoints2, descriptors2);
    //-- Step 2: Matching descriptor vectors with a FLANN based matcher
    // Since SURF is a floating-point descriptor NORM_L2 is used
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
    std::vector<std::vector<DMatch>> knn_matches;
    matcher->knnMatch(descriptors1, descriptors2, knn_matches, 2);
    //-- Filter matches using the Lowe's ratio test
    const float ratio_thresh = 0.7f;
    std::vector<DMatch> good_matches;
    for (size_t i = 0; i < knn_matches.size(); i++) {
        if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance) {
            good_matches.push_back(knn_matches[i][0]);
        }
    }

    cout << "keypoint1:" << keypoints1.size() << "\tkeypoints:" << keypoints2.size()
         << "\tgoodmatches:" << good_matches.size() << endl;
    cout << "descriptors2:" << descriptors1.size() << "\tdescriptors2:" << descriptors2.size()
         << "\tgoodmatches:" << good_matches.size() << endl;
    cout << (double) good_matches.size() / keypoints1.size();

    FileStorage fs("/Volumes/develop/data/similarity/desc1.xml", FileStorage::WRITE);

//    fs << "vocabulary" << descriptors1;
    write(fs, "data", descriptors1);

    fs.release();

    FileStorage fs1("/Volumes/develop/data/similarity/desc1.xml", FileStorage::READ);
    read(fs1["data"], descriptors1);
    fs1.release();

    //    -- Draw matches
    Mat img_matches;
    drawMatches(img1, keypoints1, img2, keypoints2, good_matches, img_matches, Scalar::all(-1),
                Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
//    -- Show detected matches
    imshow("Good Matches", img_matches);
    waitKey();
    return 0;
}


//int main() {
//    std::cout << "This tutorial code needs the xfeatures2d contrib module to be run." << std::endl;
//    return 0;
//}

#include <numeric>
#include "matching2D.hpp"

using namespace std;

// Find best matches for keypoints in two camera images based on several matching methods
void matchDescriptors(std::vector<cv::KeyPoint> &kPtsSource, std::vector<cv::KeyPoint> &kPtsRef, cv::Mat &descSource, cv::Mat &descRef,
                      std::vector<cv::DMatch> &matches, std::string descriptorType, std::string matcherType, std::string selectorType)
{
    // configure matcher
    bool crossCheck = false;
    cv::Ptr<cv::DescriptorMatcher> matcher;

    if (matcherType.compare("MAT_BF") == 0)
    {
        int normType = descriptorType.compare("DES_BINARY") == 0 ? cv::NORM_HAMMING : cv::NORM_L2;
        matcher = cv::BFMatcher::create(normType, crossCheck);
    }
    else if (matcherType.compare("MAT_FLANN") == 0)
    {
        // ...

         if (descSource.type() != CV_32F || descRef.type() != CV_32F )
        { // OpenCV bug workaround : convert binary descriptors to floating point due to a bug in current OpenCV implementation
            descSource.convertTo(descSource, CV_32F);
            descRef.convertTo(descRef, CV_32F);
        }
          matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);

    }

    // perform matching task
    if (selectorType.compare("SEL_NN") == 0)
    { // nearest neighbor (best match)

        matcher->match(descSource, descRef, matches); // Finds the best match for each descriptor in desc1
    }
    else if (selectorType.compare("SEL_KNN") == 0)
    { // k nearest neighbors (k=2)
        vector< vector<cv::DMatch> > knn_matches;
        //double t = (double)cv::getTickCount();
        matcher->knnMatch( descSource, descRef, knn_matches, 2 );
        //t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        //cout << " (KNN) with n=" << knn_matches.size() << " matches in " << 1000 * t / 1.0 << " ms" << endl;

        //-- Filter matches using the Lowe's ratio test, reference https://docs.opencv.org/3.4/d5/d6f/tutorial_feature_flann_matcher.html
        double minDescDistRatio = 0.8;
        for (auto it = knn_matches.begin(); it != knn_matches.end(); ++it)
        {

            if ((*it)[0].distance < minDescDistRatio * (*it)[1].distance)
            {
                matches.push_back((*it)[0]);
            }
        }
    }
}

// Use one of several types of state-of-art descriptors to uniquely identify keypoints
void descKeypoints(vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors, string descriptorType)
{
    // select appropriate descriptor
    cv::Ptr<cv::DescriptorExtractor> extractor;
    if (descriptorType.compare("BRISK") == 0)
    {

        int threshold = 30;        // FAST/AGAST detection threshold score.
        int octaves = 3;           // detection octaves (use 0 to do single scale)
        double patternScale = 1.0; // apply this scale to the pattern used for sampling the neighbourhood of a keypoint.

        extractor = cv::BRISK::create(threshold, octaves, patternScale);
    }
    else if(descriptorType.compare("ORB") == 0)
    {
        extractor = cv::ORB::create();
    }else if(descriptorType.compare("AKAZE") == 0)
    {
        extractor = cv::AKAZE::create();
    }else if(descriptorType.compare("BRIEF") ==0){
        extractor = cv::xfeatures2d::BriefDescriptorExtractor::create();

    }else if(descriptorType.compare("SIFT") ==0){
        extractor = cv::xfeatures2d::SIFT::create();

    }else if(descriptorType.compare("FREAK") ==0){
        extractor = cv::xfeatures2d::FREAK::create();
    }

    // perform feature description
    //double t = (double)cv::getTickCount();
    extractor->compute(img, keypoints, descriptors);
    //t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    //cout << descriptorType << " descriptor extraction in " << 1000 * t / 1.0 << " ms" << endl;
}

// Detect keypoints in image using the traditional Shi-Thomasi detector
void detKeypointsShiTomasi(vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    // compute detector parameters based on image size
    int blockSize = 4;       //  size of an average block for computing a derivative covariation matrix over each pixel neighborhood
    double maxOverlap = 0.0; // max. permissible overlap between two features in %
    double minDistance = (1.0 - maxOverlap) * blockSize;
    int maxCorners = img.rows * img.cols / max(1.0, minDistance); // max. num. of keypoints

    double qualityLevel = 0.01; // minimal accepted quality of image corners
    double k = 0.04;

    // Apply corner detection
    //double t = (double)cv::getTickCount();
    vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(img, corners, maxCorners, qualityLevel, minDistance, cv::Mat(), blockSize, false, k);

    // add corners to result vector
    for (auto it = corners.begin(); it != corners.end(); ++it)
    {

        cv::KeyPoint newKeyPoint;
        newKeyPoint.pt = cv::Point2f((*it).x, (*it).y);
        newKeyPoint.size = blockSize;
        keypoints.push_back(newKeyPoint);
    }
    //t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    //cout << "Shi-Tomasi detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Shi-Tomasi Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}


// Detect keypoints in image using the traditional Shi-Thomasi detector
void detKeypointsHarris(vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{   
    // Detector parameters
    int blockSize = 2;     // for every pixel, a blockSize × blockSize neighborhood is considered
    int apertureSize = 3;  // aperture parameter for Sobel operator (must be odd)
    int minResponse = 100; // minimum value for a corner in the 8bit scaled response matrix
    double k = 0.04;       // Harris parameter (see equation for details)

    // Detect Harris corners and normalize output
    cv::Mat dst, dst_norm, dst_norm_scaled;
    dst = cv::Mat::zeros(img.size(), CV_32FC1);
    // Apply corner detection

    //double t = (double)cv::getTickCount();
    cv::cornerHarris(img, dst, blockSize, apertureSize, k, cv::BORDER_DEFAULT);
    cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
    cv::convertScaleAbs(dst_norm, dst_norm_scaled);
    //const int sw_size = 7;
    //const int sw_dist = sw_size / 2;
    const int windowSize = apertureSize * 2 + 1;
    const int col = dst_norm_scaled.cols;
    const int row = dst_norm_scaled.rows;

    // iterate all the points inside the boundery
    if(col >= windowSize && row >= windowSize){ // the image row and col size should at least greater that windowSize

        for(int ci = apertureSize; ci<row-apertureSize-1; ci++){
            for(int cj = apertureSize; cj<col-apertureSize-1; cj++){  // iterate all the central pixels (i,j)

                //define the keyPoint value which is equal the current one
                bool stop = false;
                unsigned char min_val = static_cast<unsigned char>(minResponse);
                unsigned char max_val = dst_norm_scaled.at<unsigned char>(ci,cj);
                if(dst_norm_scaled.at<unsigned char>(ci,cj) > min_val)
                {
                    for(int ni = ci-apertureSize; ni < ci+apertureSize; ni++){
                        for(int nj = cj- apertureSize ; nj<cj + apertureSize; nj++){
                            // iterate all the points in the window

                            // if  one of the response > central point, record this value and break, otherwise continue
                            if( dst_norm_scaled.at<unsigned char>(ni,nj)>max_val  ){

                                max_val = dst_norm_scaled.at<unsigned char>(ni,nj);
                                stop = true;
                                break;
                            }
                        }
                        if(stop){
                            break;
                        }
                    }
                    if(dst_norm_scaled.at<unsigned char>(ci,cj) == max_val){
                        keypoints.emplace_back(cj, ci , 2 * apertureSize, -1, max_val);
                    }
                }
            }
        }

    }

    //t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    //cout << "Harris detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Harris Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}


// Detect keypoints in image using the traditional Shi-Thomasi detector
void detKeypointsModern(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, std::string detectorType, bool bVis)
{
    //create  a detector
    // reference from :
    cv::Ptr<cv::FeatureDetector> detector;

    if(detectorType == "BRISK"){
        detector = cv::BRISK::create();
    }else if(detectorType == "ORB"){
        detector = cv::ORB::create();

    }else if(detectorType == "AKAZE"){
        detector = cv::AKAZE::create();
        
    }else if(detectorType == "FAST"){
        const int threshold = 30;
        const bool bNMS = true;
        detector = cv::FastFeatureDetector::create(threshold,bNMS,cv::FastFeatureDetector::TYPE_9_16);
    }else if(detectorType == "SIFT"){
        detector = cv::xfeatures2d::SIFT::create();
    }


    //double t = (double)cv::getTickCount();
    detector->detect(img, keypoints);
    //t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    //cout << "Modern detector "<< detectorType<< " with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Modern Corner Detector " + detectorType +  " Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }


}


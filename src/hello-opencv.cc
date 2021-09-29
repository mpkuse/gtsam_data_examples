#include <iostream>
#include <opencv2/opencv.hpp>

#include "utils/CvPlottingUtils.h"
int main()
{
    std::cout << "Hello OpenCV\n";
    cv::Mat test_im = cv::imread( "../data/rgb-d/rgb/1.png" ); 
    std::cout << "test_im: " << CvPlottingUtils::cvmat_info( test_im ) << std::endl;
    cv::imshow( "test_im", test_im );
    cv::waitKey(0);
}
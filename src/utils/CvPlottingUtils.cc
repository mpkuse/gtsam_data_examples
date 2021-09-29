#include "CvPlottingUtils.h"


void CvPlottingUtils::say_hi()
{
    std::cout << "Hello say hi\n";
}

std::string CvPlottingUtils::type2str(int type) {
    std::string r;

    uchar depth = type & CV_MAT_DEPTH_MASK;
    uchar chans = 1 + (type >> CV_CN_SHIFT);

    switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
    }

    r += "C";
    r += (chans+'0');

    return r;
}

std::string CvPlottingUtils::cvmat_info( const cv::Mat& mat )
{
    std::stringstream buffer;
    buffer << "shape=" << mat.rows << "," << mat.cols << "," << mat.channels() ;
    buffer << "\t" << "dtype=" << CvPlottingUtils::type2str( mat.type() );
    return buffer.str();
}

std::string CvPlottingUtils::eigenmat_info( const Eigen::MatrixXd& mat )
{
    std::stringstream buffer;
    buffer << "shape=" << mat.rows() << "," << mat.cols() << " ";
    return buffer.str();
}

std::vector<std::string>
CvPlottingUtils::split( std::string const& original, char separator )
{
    std::vector<std::string> results;
    std::string::const_iterator start = original.begin();
    std::string::const_iterator end = original.end();
    std::string::const_iterator next = std::find( start, end, separator );
    while ( next != end ) {
        results.push_back( std::string( start, next ) );
        start = next + 1;
        next = std::find( start, end, separator );
    }
    results.push_back( std::string( start, next ) );
    return results;
}

void CvPlottingUtils::keypoint_2_eigen( const std::vector<cv::KeyPoint>& kp, Eigen::MatrixXd& uv, bool make_homogeneous )
{
    EXPECT_GT( kp.size(), 0 ) << "keypoints cannot be empty\n";
    uv = Eigen::MatrixXd::Constant( (make_homogeneous?3:2), kp.size(), 1.0 );
    for( int i=0; i<kp.size() ; i++ )
    {
        uv(0,i) = kp.at(i).pt.x;
        uv(1,i) = kp.at(i).pt.y;
    }
}

void CvPlottingUtils::keypoint_2_eigen( const std::vector<cv::Point2f>& kp, Eigen::MatrixXd& uv, bool make_homogeneous )
{
    EXPECT_GT( kp.size(), 0 ) << "keypoints cannot be empty\n";
    uv = Eigen::MatrixXd::Constant( (make_homogeneous?3:2), kp.size(), 1.0 );
    for( int i=0; i<kp.size() ; i++ )
    {
        uv(0,i) = kp.at(i).x;
        uv(1,i) = kp.at(i).y;
    }
}

void CvPlottingUtils::eigen_2_keypoint( const Eigen::MatrixXd& uv, std::vector<cv::Point2f>& kp )
{
    EXPECT_GT( uv.cols(), 0 );
    EXPECT_TRUE( uv.rows()==2 || uv.rows() == 3 );
    kp.clear(); 
    for( int i=0 ; i<uv.cols() ; i++ )
    {
        kp.push_back( cv::Point2f( uv(0,i), uv(1,i)  ) );
    }
}

void CvPlottingUtils::dmatch_2_eigen( const std::vector<cv::KeyPoint>& kp1, const std::vector<cv::KeyPoint>& kp2,
                            const std::vector<cv::DMatch> matches,
                            Eigen::MatrixXd& M1, Eigen::MatrixXd& M2,
                            bool make_homogeneous
                        )
{
    EXPECT_GT( matches.size(), 0 );
    EXPECT_GT( kp1.size() , 0 );
    EXPECT_GT( kp2.size() , 0 );

    M1 = Eigen::MatrixXd::Constant( (make_homogeneous?3:2), matches.size(), 1.0 );
    M2 = Eigen::MatrixXd::Constant( (make_homogeneous?3:2), matches.size(), 1.0 );
    for( int i=0 ; i<matches.size() ; i++ ) {
        int queryIdx = matches.at(i).queryIdx; //kp1
        int trainIdx = matches.at(i).trainIdx; //kp2
        // assert( queryIdx >=0 && queryIdx < kp1.size() );
        // assert( trainIdx >=0 && trainIdx < kp2.size() );
        EXPECT_GE( queryIdx, 0); EXPECT_LT( queryIdx, kp1.size() );
        EXPECT_GE( trainIdx, 0); EXPECT_LT( trainIdx, kp2.size() );
        M1(0,i) = kp1.at(queryIdx).pt.x;
        M1(1,i) = kp1.at(queryIdx).pt.y;

        M2(0,i) = kp2.at(trainIdx).pt.x;
        M2(1,i) = kp2.at(trainIdx).pt.y;
    }
}

void CvPlottingUtils::dmatch_2_cvpoint2f( const std::vector<cv::KeyPoint>& kp1, const std::vector<cv::KeyPoint>& kp2,
                                const std::vector<cv::DMatch> matches,
                                std::vector<cv::Point2f>& mp1, std::vector<cv::Point2f>& mp2
                            )
{
    EXPECT_GT( matches.size(), 0 );
    EXPECT_GT( kp1.size() , 0 );
    EXPECT_GT( kp2.size() , 0 );

    // M1 = Eigen::MatrixXd::Constant( (make_homogeneous?3:2), matches.size(), 1.0 );
    // M2 = Eigen::MatrixXd::Constant( (make_homogeneous?3:2), matches.size(), 1.0 );
    mp1.clear();
    mp2.clear(); 
    for( int i=0 ; i<matches.size() ; i++ ) {
        int queryIdx = matches.at(i).queryIdx; //kp1
        int trainIdx = matches.at(i).trainIdx; //kp2
        // assert( queryIdx >=0 && queryIdx < kp1.size() );
        // assert( trainIdx >=0 && trainIdx < kp2.size() );
        EXPECT_GE( queryIdx, 0); EXPECT_LT( queryIdx, kp1.size() );
        EXPECT_GE( trainIdx, 0); EXPECT_LT( trainIdx, kp2.size() );
        float _p1_x = kp1.at(queryIdx).pt.x;
        float _p1_y = kp1.at(queryIdx).pt.y;

        float _p2_x = kp2.at(trainIdx).pt.x;
        float _p2_y = kp2.at(trainIdx).pt.y;
        
        mp1.push_back( cv::Point2f(_p1_x,_p1_y) );
        mp2.push_back( cv::Point2f(_p2_x, _p2_y) );
    }

    EXPECT_EQ( mp1.size(), mp2.size() );    
}                            



bool CvPlottingUtils::vec_of_floats_2_eigen( const std::vector<float>& d_pts, Eigen::VectorXd& d_pts_eigen )
{
    EXPECT_GT( d_pts.size(), 0 );
    if( d_pts.size() == 0 )
        return false; 

    d_pts_eigen = Eigen::VectorXd::Zero( d_pts.size() );    
    for(int i=0 ; i<d_pts.size() ; i++ ) {
        d_pts_eigen(i) = d_pts.at(i);
    }
    return true; 
}

bool CvPlottingUtils::eigen_2_vec_of_floats( const Eigen::VectorXd& d_pts_eigen, std::vector<float>& d_pts )
{
    d_pts.clear();
    for( int i=0 ; i< d_pts_eigen.size() ; i++ )
    {
        d_pts.push_back( (float)d_pts_eigen(i)  );
    }
    return true; 
}


void CvPlottingUtils::plot_point_sets( 
    cv::Mat& im, 
    const Eigen::MatrixXd& pts_set, 
    cv::Mat& dst,
    
    const cv::Scalar& color, 
    const std::string& msg, 
    bool enable_anotations,
    const Eigen::VectorXi& annotations,
    const std::vector<bool>& mask
)
{
  
    EXPECT_FALSE( im.empty() );
    EXPECT_GT( pts_set.cols() , 0 );
    EXPECT_NE( im.data, dst.data );


    int n_annotations = annotations.size(); 
    if( enable_anotations )
    {
        if( n_annotations > 0 )
            EXPECT_EQ(  pts_set.cols(), n_annotations );
    }

    EXPECT_TRUE( mask.size() == 0 || mask.size() == pts_set.cols() );
    
    
    if( im.channels() == 1 )
    cv::cvtColor( im, dst, cv::COLOR_GRAY2BGR );
    else
    im.copyTo(dst);


  // cv::putText( dst, to_string(msg.length()), cv::Point(5,5), cv::FONT_HERSHEY_COMPLEX_SMALL, .95, cv::Scalar(0,255,255) );
  if( msg.length() > 0 ) {
    std::vector<std::string> msg_split;
    msg_split = split( msg, ';' );
    for( int q=0 ; q<msg_split.size() ; q++ )
      cv::putText( dst, msg_split[q], cv::Point(5,20+20*q), cv::FONT_HERSHEY_COMPLEX_SMALL, .95, cv::Scalar(0,255,255) );
  }


  //pts_set is 2xN
  cv::Point2d pt;
  for( int i=0 ; i<pts_set.cols() ; i++ )
  {
    if( mask.size() > 0 )
        if( !mask.at(i)  )
            continue; 

    // pt = cv::Point2d(pts_set.at<float>(0,i),pts_set.at<float>(1,i) );
    pt = cv::Point2d( (float)pts_set(0,i), (float)pts_set(1,i) );
    cv::circle( dst, pt, 2, color, -1 );

    if( enable_anotations ) {
        // char to_s[20];
        // sprintf( to_s, "%d", annotations(i) );
        std::string to_s; 
        if( n_annotations > 0 )
            to_s = std::to_string( annotations(i) );
        else 
            to_s = std::to_string( i );
        cv::putText( dst, to_s, pt, cv::FONT_HERSHEY_COMPLEX_SMALL, 0.5, color  );
    }
  }
}


void CvPlottingUtils::plot_point_pair( 
                    const cv::Mat& imA, const Eigen::MatrixXd& ptsA, int idxA,
                    const cv::Mat& imB, const Eigen::MatrixXd& ptsB, int idxB,
                    cv::Mat& dst,
                    const cv::Scalar& color_marker,
                    const cv::Scalar& color_line,
                      bool annotate_pts,
                      const std::string& msg
                    )
{
    // ptsA : ptsB : 2xN or 3xN

    EXPECT_FALSE( imA.empty() );
    EXPECT_FALSE( imB.empty() );
    EXPECT_TRUE( imA.rows == imB.rows && imA.rows > 0  );
    EXPECT_TRUE( imA.cols == imB.cols && imB.cols > 0  );
    EXPECT_TRUE( imA.channels() == imB.channels() );
    EXPECT_TRUE( ptsA.cols() == ptsB.cols() && ptsA.cols() > 0 );

    cv::Mat outImg_;
    cv::hconcat(imA, imB, outImg_);

    cv::Mat outImg;
    if( outImg_.channels() == 3 )
        outImg = outImg_;
    else
        cv::cvtColor( outImg_, outImg,  cv::COLOR_GRAY2BGR );




    // loop over all points
    int count = 0;
    for( int kl=0 ; kl<ptsA.cols() ; kl++ )
    {
        // if( mask(kl) == 0 )
        //   continue;

        count++;
        cv::Point2d A( ptsA(0,kl), ptsA(1,kl) );
        cv::Point2d B( ptsB(0,kl), ptsB(1,kl) );

        cv::circle( outImg, A, 3,color_marker, -1 );
        cv::circle( outImg, B+cv::Point2d(imA.cols,0), 3,color_marker, -1 );

        cv::line( outImg,  A, B+cv::Point2d(imA.cols,0), color_line );

        if( annotate_pts )
        {
            cv::putText( outImg, std::to_string(kl), A, cv::FONT_HERSHEY_SIMPLEX, 0.3, color_marker, 1 );
            cv::putText( outImg, std::to_string(kl), B+cv::Point2d(imA.cols,0), cv::FONT_HERSHEY_SIMPLEX, 0.3, color_marker, 1 );
        }
    }

    // if( msg.length() > 0 )
    std::vector<std::string> msg_tokens;
    if( msg.length() > 0 )
        msg_tokens = split(msg, ';');
    int status_img_len = 100 + msg_tokens.size() * 20; 

    cv::Mat status = cv::Mat(status_img_len, outImg.cols, CV_8UC3, cv::Scalar(0,0,0) );
    cv::putText( status, std::to_string(idxA).c_str(), cv::Point(10,30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255,255,255), 2 );
    cv::putText( status, std::to_string(idxB).c_str(), cv::Point(imA.cols+10,30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255,255,255), 2 );
    cv::putText( status, "marked # pts: "+std::to_string(count), cv::Point(10,60), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255,255,255), 1.5 );


    // put msg in status image
    if( msg.length() > 0 ) { // ':' separated. Each will go in new line
        
        for( int h=0 ; h<msg_tokens.size() ; h++ )
            cv::putText( status, msg_tokens[h].c_str(), cv::Point(10,80+20*h), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255,255,255), 1.5 );
    }


    cv::vconcat( outImg, status, dst );

}

#include <opencv2/opencv.hpp>
#include <opencv2/opencv_modules.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/flann/flann.hpp>

#include <time.h>
#include <math.h>
#include <iostream>
#include <string>
#include <algorithm>
#include <iterator>
#include <vector>

using namespace cv;

void zoom_the_image();
Mat get_homography(Mat& image);
void feature_detection();
void super_resolution_interpolation(std::vector<Mat> H);
void calculate_PSNR();

extern void cuda_doStuff(Mat* img,
						 Mat& result,
						 std::vector<Mat>* images, 
						 float* remappedX, 
						 float* remappedY, 
						 float* minimum,
						 float* maximum,
						 std::vector<Mat> H);

/** Global variables
    Images used in all functions and main()
**/
// new blank images
Mat high_res_img;
Mat high_res_img2;
Mat zoomed_img;
Mat zoomed_img2;
Mat low_res_img;
Mat virtual_img;
Mat result_img1;
Mat warped_img;
Mat final_result;
Mat horizontal_gradient_img;
Mat vertical_gradient_img;

Size image_size;
Size virtual_size;

std::string low_image_name = "images/castle.png";
std::string zoom_image_name = "images/castle.png";

// creates a new zoomed image from the low resolution image by cropping it
void zoom_the_image() {

    if(virtual_size.height > image_size.height/4 && virtual_size.width > image_size.width/4) {
        virtual_size.width -= 32;
        virtual_size.height -= 32;
        int shift_in_x = (image_size.width-virtual_size.width)/2;
        int shift_in_y = (image_size.height-virtual_size.height)/2;
        std::cout << virtual_size << std::endl;
        std::cout << shift_in_x << " " << shift_in_y << std::endl;

        Rect new_size(shift_in_x, shift_in_y, virtual_size.width, virtual_size.height);
        virtual_img = low_res_img(new_size);

        resize(virtual_img, virtual_img, image_size);
    }

}

Mat get_homography(Mat& image) {

    std::vector<KeyPoint> keypoints_1;
    std::vector<KeyPoint> keypoints_2;


//    Ptr<ORB> detector = ORB::create(5000,
//                                    1.3f,
//                                    8,
//                                    32,
//                                    0,
//                                    2,
//                                    ORB::HARRIS_SCORE,
//                                    15,
//                                    16);

    Ptr<xfeatures2d::SIFT> detector = xfeatures2d::SIFT::create();

    //  detects the keypoints from the images
    detector->detect(virtual_img, keypoints_1);
    detector->detect(image, keypoints_2);
    std::cout << keypoints_1[0].pt.x << " " << keypoints_2[0].pt.x << std::endl;

    Mat descriptor_1;
    Mat descriptor_2;

    //  finds the feature vectors from the image
    //Ptr<xfeatures2d::FREAK> extractor = xfeatures2d::FREAK::create();
    //extractor->compute(virtual_img, keypoints_1, descriptor_1);
    //extractor->compute(zoomed_img, keypoints_2, descriptor_2);
    detector->compute(virtual_img, keypoints_1, descriptor_1);
    detector->compute(image, keypoints_2, descriptor_2);

    //BFMatcher bruteForceMatcher(NORM_HAMMING);
    BFMatcher bruteForceMatcher(NORM_L2, false);
    //FlannBasedMatcher bruteForceMatcher;
    std::vector< DMatch > matches;



    //  matches the feature vectors using a brute force matcher
    bruteForceMatcher.match(descriptor_1, descriptor_2, matches);

    double min_dist = 100;
    double max_dist = 0;
    double dist;

    //  extract the good matches with distances below 3*minimum distance
    for(int i = 0; i < matches.size(); i++) {
        dist = matches[i].distance;
        if( dist > max_dist && dist > 0) max_dist = dist;
        if( dist < min_dist && dist > 0) min_dist = dist;
    }
    std::cout << "match max dist: " << max_dist << ", min dist: " << min_dist << std::endl;
    std::vector< DMatch > good_matches;
    for(int i = 0; i < matches.size(); i++) {
        if( matches[i].distance < (min_dist+max_dist)/2 ) {
            good_matches.push_back( matches[i] );
        }
    }

    drawMatches(virtual_img, keypoints_1, image, keypoints_2, good_matches, result_img1);

    //  create matching feature points
    std::vector< Point2f > obj;
    std::vector< Point2f > scene;
    for(int i = 0; i < good_matches.size(); i++) {
        scene.push_back( keypoints_1[good_matches[i].queryIdx].pt );
        obj.push_back( keypoints_2[good_matches[i].trainIdx].pt );
    }
    // create a homography matrix H, warp the zoomed image with it and place the image to result
    Mat H = findHomography( obj, scene, CV_RANSAC );

    warpPerspective(image, warped_img, H, image_size, INTER_NEAREST);

    // corners:
    //  0 = top-left
    //  1 = top-right
    //  2 = bottom-right
    //  3 = bottom-left

    std::vector<Point2f> obj_corners(4);
    obj_corners[0] = cvPoint(0,0); obj_corners[1] = cvPoint( image.cols, 0 );
    obj_corners[2] = cvPoint( image.cols, image.rows ); obj_corners[3] = cvPoint( 0, image.rows );
    std::vector<Point2f> scene_corners(4);

    perspectiveTransform( obj_corners, scene_corners, H);

    std::cout << scene_corners << std::endl;

    line( result_img1, scene_corners[0], scene_corners[1], Scalar( 0, 255, 0), 4 );
    line( result_img1, scene_corners[1], scene_corners[2], Scalar( 0, 255, 0), 4 );
    line( result_img1, scene_corners[2], scene_corners[3], Scalar( 0, 255, 0), 4 );
    line( result_img1, scene_corners[3], scene_corners[0], Scalar( 0, 255, 0), 4 );

    imshow("Brute force matching", result_img1);
    imshow("Warped image", warped_img);
    waitKey(0);

    H.convertTo(H, CV_32FC1);
    return H;

}

void feature_detection() {

    clock_t timer;
    timer = clock();

    std::vector<Mat> Hs;
    Hs.push_back(get_homography(zoomed_img));
    //Hs.push_back(get_homography(zoomed_img2));

    super_resolution_interpolation(Hs);

    timer = clock() - timer;
    std::cout << "Time taken: " << (float)timer/1000 << " seconds" << std::endl;

    calculate_PSNR();
    imshow("virtual image", final_result);
    waitKey(0);
    destroyAllWindows();
}

bool lessThanPointsX(Point2f & pnt, float max_val) {
    //std::cout << left.y << " " << right << std::endl;
    return (max_val > pnt.x);
}
bool moreThanPointsX(float min_val, Point2f & pnt) {
    //std::cout << right.y << " " << left << std::endl;
    return (min_val < pnt.x);
}
bool lessThanPointsY(Point2f & pnt, float max_val) {
    //std::cout << left.y << " " << right << std::endl;
    return (max_val > pnt.y);
}
bool moreThanPointsY(float min_val, Point2f & pnt) {
    //std::cout << right.y << " " << left << std::endl;
    return (min_val < pnt.y);
}

void super_resolution_interpolation(std::vector<Mat> H) {

    std::cout << "superresolution in progress..." << std::endl;

    final_result = Mat::zeros(image_size, CV_8UC1);

//    int pad = 2;
//    Mat padded_low = Mat::zeros(Size(virtual_size.width+pad*2,virtual_size.height+pad*2), CV_8UC1);
//    virtual_img.copyTo(padded_low(Rect(pad, pad, virtual_img.cols, virtual_img.rows)));
//    int closest_px[2];

    std::vector<Mat> images;
    images.push_back(zoomed_img);
    //images.push_back(zoomed_img2);

    Mat mat1(3,1, CV_32FC1);
    Mat mat2;

	float* remappedX = new float[image_size.width*image_size.height];
	float* remappedY = new float[image_size.width*image_size.height];
	//remappedX.reserve(image_size.width*image_size.height*sizeof(float));
	//remappedY.reserve(image_size.width*image_size.height*sizeof(float));
	float new_x;
    float new_y;
    float maximums[2] = {0,0};
    float minimums[2] = {1000000,1000000};
    //  create new coordinates: normal (integer) -> subpixels (float)
    for(int i = 0; i < images.size(); i++) {
        for(int x = 0; x < images[i].size().width; x++) {
			for (int y = 0; y < images[i].size().height; y++) {
				mat1.at<float>(0) = x;
				mat1.at<float>(1) = y;
				mat1.at<float>(2) = 1.0;

				mat2 = H[i] * mat1;
				new_x = mat2.at<float>(0) / mat2.at<float>(2);
				new_y = mat2.at<float>(1) / mat2.at<float>(2);
				//std::cout << x + y*images[i].size().width << " " << new_x << std::endl;
				remappedX[x + y*images[i].size().width] = new_x;
				remappedY[x + y*images[i].size().width] = new_y;
                if(new_x > maximums[0]) {
                    maximums[0] = new_x;
                }
                if(new_x < minimums[0]) {
                    minimums[0] = new_x;
                }
                if(new_y > maximums[1]) {
                    maximums[1] = new_y;
                }
                if(new_y < minimums[1]) {
                    minimums[1] = new_y;
                }
            }
        }
    }

    std::cout << remappedX[256*255] << std::endl;
    //std::cout << remapped[0][images[0].size().height*(images[0].size().width-1)] << std::endl;
    //std::cout << remapped[0][images[0].size().height*images[0].size().width-1] << std::endl;
    //std::cout << remapped[0][images[0].size().height-1] << std::endl;

	cuda_doStuff(&virtual_img, final_result, &images, remappedX, remappedY, minimums, maximums, H);

    //local variables
    float scale[] = {(float)image_size.width/(float)virtual_size.width,
                     (float)image_size.height/(float)virtual_size.height};
    float distance;
    float weight;
    float a = -0.5;
    float average;
    float total_weight;
    float search_range = 0.75*(scale[0]);
    std::vector< float > subpixels;
    std::vector< float > weights;
    //std::cout << 1/scale[0] << " " << 1/scale[1] << " " << search_range << std::endl;

    ////  go trough every pixel of the area found by matching the zoomed image
    //for(int x = 0; x < image_size.width; x++) {
    //    //std::cout << x << ", ";
    //    for(int y = 0; y < image_size.height; y++) {
    //        //std::cout << x << " " << y << std::endl;

    //        average = 0.0;
    //        total_weight = 0.0;

    //        subpixels.push_back(virtual_img.at<uchar>(y,x));
    //        weights.push_back(1);

    //        /*closest_px[0] = ceil((float)x/scale[0]);
    //        closest_px[1] = ceil((float)y/scale[1]);
    //        for(int col = -2; col < 2; col++) {
    //            for(int row = -2; row < 2; row++) {
    //                subpixels.push_back(padded_low.at<uchar>(closest_px[1]+col+2, closest_px[0]+row+2));
    //                distance = sqrtf(pow(x/2-closest_px[0]+row,2) + pow(y/2-closest_px[1]+col,2));
    //                if(distance <= 1) {
    //                    weight = (a+2)*pow(distance, 3) - (a+3)*pow(distance, 2) + 1;
    //                }
    //                else if (distance > 1 && distance < 2) {
    //                    weight = a*pow(distance, 3) - 5*a*pow(distance, 2) + 8*a*distance - 4*a;
    //                }
    //                else weight = 0;
    //                weights.push_back(weight);
    //            }
    //        }*/
    //        for(int j = 0; j < images.size(); j++ ) {
    //            if( x >= minimums[j][0] && y >= minimums[j][1] &&
    //                x <= maximums[j][0] && y <= maximums[j][1]) {
    //                int w = images[j].size().height;
    //                std::vector<Point2f>::iterator it;
    //                auto lower = std::lower_bound(remapped[j].begin(), remapped[j].end(), x-search_range, lessThanPointsX);
    //                auto upper = std::upper_bound(remapped[j].begin(), remapped[j].end(), x+search_range, moreThanPointsX);
    //                //auto lower = std::lower_bound(remapped[j].begin(), remapped[j].end(), y-search_range, lessThanPointsY);
    //                //auto upper = std::upper_bound(remapped[j].begin(), remapped[j].end(), y+search_range, moreThanPointsY);
    //                //std::cout << y << "  lower: " << *lower << "  upper: " << *(upper) << std::endl;
    //                //int low_idx = max(std::distance(remapped[j].begin(), lowerX), std::distance(remapped[j].begin(), lowerY));
    //                //int upp_idx = min(std::distance(remapped[j].begin(), upperX), std::distance(remapped[j].begin(), upperY));
    //                int low_idx = std::distance(remapped[j].begin(), lower);
    //                int upp_idx = std::distance(remapped[j].begin(), upper);
    //                for(int k = low_idx; k != upp_idx; k++) {
    //                    distance = ( pow(remapped[j][k].x-x, 2) +
    //                                 pow(remapped[j][k].y-y, 2));
    //                    weight = 1/distance;
    //                    subpixels.push_back(images[j].at<uchar>(k%w,k/w));
    //                    weights.push_back(weight);
    //                    /*float dx;
    //                    float dy;
    //                    float Wf;
    //                    float Gf;
    //                    float mu = -0.9;
    //                    float m = 2;
    //                    dx = 1 - abs(remapped[j][k].x-x);
    //                    dy = 1 - abs(remapped[j][k].y-y);
    //                    if(abs(dx) < 2*search_range && abs(dy) < 2*search_range) {
    //                        Gf = ((abs(horizontal_gradient_img.at<float>(k%w,k/w)) + abs(vertical_gradient_img.at<float>(k%w,k/w))) /
    //                              (2*sqrt(pow(horizontal_gradient_img.at<float>(k%w,k/w),2)+ pow(vertical_gradient_img.at<float>(k%w,k/w),2))));
    //                        if (std::isnan(Gf)) Gf = 1000000;
    //                        Wf = pow(mu*Gf+1, m);
    //                        weight = dx*dy*Wf;
    //                        subpixels.push_back(images[j].at<uchar>(k%w,k/w));
    //                        weights.push_back(weight);
    //                    }*/
    //                }
    //            }
    //        }
    //        //  go through the zoomed image and take every subpixel that is less than
    //        //  some distance away from the real pixel in result image
    //        /*for(int j = 0; j < images.size(); j++ ) {
    //            if( x >= minimums[j][0] && y >= minimums[j][1]) {
    //                int xx,yy;
    //                int w = images[j].size().width;
    //                for(xx = 0; xx < images[j].size().height; xx++) {
    //                    if(remapped[j][xx+w*yy].x > x+search_range) break;

    //                    for(yy = 0; yy < w; yy++) {
    //                        if(remapped[j][xx+w*yy].y > y+search_range) break;

    //                        if( remapped[j][xx+w*yy].x <= x+search_range &&
    //                            remapped[j][xx+w*yy].x >= x-search_range &&
    //                            remapped[j][xx+w*yy].y <= y+search_range &&
    //                            remapped[j][xx+w*yy].y >= y-search_range ) {

    //                            distance = sqrtf( pow(remapped[j][xx+w*yy].x-x, 2) +
    //                                              pow(remapped[j][xx+w*yy].y-y, 2));
    ////                            if(distance < 1) {
    ////                                weight = (a+2)*pow(distance, 3) - (a+3)*pow(distance, 2) + 1;
    ////                            }
    ////                            else if (distance > 1 && distance < 2) {
    ////                                weight = a*pow(distance, 3) - 5*a*pow(distance, 2) + 8*a*distance - 4*a;
    ////                            }
    ////                            else weight = 0;
    //                            weight = 1/distance;

    //                            //std::cout << x << " " << y << "  " << remapped[xx+image_size.width*yy].x << " " << remapped[xx+image_size.width*yy].y << std::endl;
    //                            subpixels.push_back(images[j].at<uchar>(xx,yy));
    //                            weights.push_back(weight);
    //                        }
    //                    }
    //                }
    //            }
    //        }*/
    //        //  calculate the total weights of the pixels
    //        for(int j = 0; j < weights.size(); j++) {
    //            total_weight += weights[j];
    //        }

    //        //  count the average of the subpixels and set it as the result pixel
    //        if( subpixels.size() > 0 ) {
    //            int i;
    //            for(i = 0; i < subpixels.size(); i++) {
    //                //std::cout << subpixels[i].x << std::endl;
    //                average += subpixels[i]*weights[i];
    //            }
    //            //std::cout << x << " " << y << " " << i << " " << average << " " << total_weight << std::endl;
    //            final_result.at<uchar>(y,x) = (int)(average/total_weight);

    //            weights.clear();
    //            subpixels.clear();
    //        }
    //    }
    //}

     //  deblurring
//    Mat lap_img;
//    Laplacian(final_result, lap_img, CV_8UC1);
//    final_result -= lap_img/2;


}

// ONLY WORKS WHEN NOT ZOOMING
// calculates the peak signal-to-noise ration between the high resolution image and
// the virtual image created from low resolution image and the zoomed image
void calculate_PSNR() {

    Mat img1;
    Mat img2;
    Mat img3 = Mat::zeros(image_size, CV_16S);

    Rect crop((image_size.width-virtual_size.width)/2,
              (image_size.height-virtual_size.height)/2,
              (image_size.width+virtual_size.width)/2,
              (image_size.height+virtual_size.height)/2);
    img1 = high_res_img(crop);
    resize(img1, img1, image_size);

    img1.convertTo(img1, CV_16S);
    virtual_img.convertTo(img2, CV_16S);

    Scalar mean_value;
    pow((img1 - img2), 2, img3);
    mean_value = mean(img3);
    float MSE = mean_value.val[0];

    float PSNR = 10*log10((255*255)/MSE);

    std::cout << "PSNR between virtual and high res images: " << PSNR << "dB" << std::endl;

}

int main()

{

    /** Initialization of variables
    **/

    high_res_img = imread(low_image_name, 0); // read the high resolution file
    high_res_img2 = imread(zoom_image_name, 0); // read the high resolution file

    image_size = high_res_img.size();
    virtual_size = image_size;

    //  create the low resolution image by resizing the high resolution image to half and
    //  resizing it back to original size
    resize(high_res_img, low_res_img, image_size/2);
    resize(low_res_img, low_res_img, image_size);

    //  oversample the low resolution image
//    Mat inter_img = Mat::zeros(image_size, CV_8UC1);
//    for(int x = 0; x < image_size.width/2; x++) {
//        for(int y = 0; y < image_size.height/2; y++) {
//            inter_img.at<uchar>(y*2, x*2) = low_res_img.at<uchar>(y,x);
//        }
//    }
//    low_res_img = inter_img;

//    // add noise to low quality image
//    Mat noise = Mat(image_size, CV_8UC1);
//    randn(noise, 0, 25);
//    low_res_img = low_res_img + noise;

    Rect crop1(image_size.width/5, image_size.height/5, image_size.width/2, image_size.height/2);
    zoomed_img = high_res_img(crop1).clone();

    Rect crop2(2*image_size.width/5, 2*image_size.height/5,
               image_size.width/2, image_size.height/2);
    zoomed_img2 = high_res_img2(crop2).clone();

    final_result = Mat::zeros(image_size, CV_8UC1);

    Mat kernel = Mat::zeros( 3, 3, CV_32F);
	kernel.at<float>(0, 0) = -1; kernel.at<float>(0, 1) = 0; kernel.at<float>(0, 2) = 1;
	kernel.at<float>(1, 0) = -2; kernel.at<float>(1, 1) = 0; kernel.at<float>(1, 2) = 2;
	kernel.at<float>(2, 0) = -1; kernel.at<float>(2, 1) = 0; kernel.at<float>(2, 2) = 1;
    filter2D(zoomed_img, horizontal_gradient_img, -1, kernel);

	kernel.at<float>(0, 0) = -1; kernel.at<float>(0, 1) = -2; kernel.at<float>(0, 2) = -1;
	kernel.at<float>(1, 0) = 0; kernel.at<float>(1, 1) = 0; kernel.at<float>(1, 2) = 0;
	kernel.at<float>(2, 0) = 1; kernel.at<float>(2, 1) = 2; kernel.at<float>(2, 2) = 1;
    filter2D(zoomed_img, vertical_gradient_img, -1, kernel);

    namedWindow( "virtual zoom", CV_WINDOW_AUTOSIZE );
    //namedWindow( "low window", CV_WINDOW_AUTOSIZE );
    //namedWindow( "zoomed window", CV_WINDOW_AUTOSIZE );


    virtual_img = low_res_img;
    //zoomed_img.copyTo(virtual_img(crop));

    std::cout << "high " << high_res_img.size() << std::endl;
    std::cout << "low " << low_res_img.size() << std::endl;
    std::cout << "zoom " << zoomed_img.size() << std::endl;
    std::cout << "zoom2 " << zoomed_img2.size() << std::endl;
    std::cout << "virtual " << virtual_size << std::endl;

    /** Main loop.
        Shows the image and waits for input in a loop.
        Inputs:
            Esc = close the program
            'f' = start feature detection
            '+' = zoom in
    **/
    int k;
    //imshow("test imshow", test_img);
    while(1) {
        imshow("virtual zoom", virtual_img);
        imshow("high resolution", high_res_img);
        k = waitKey(10);
        if( k == 27) {
            break;
        }
        else if( k == 43 ) {
            zoom_the_image();
        }
        else if( k == 102 ) {
            feature_detection();
        }
        else {
            //do nothing
        }
    }

    return 0;

}

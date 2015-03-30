#include "opencv2/opencv.hpp"
#include <stdio.h>
#include <iostream>
#include <vector>

#define DEBUG 1

#define WINDOW_INPUT "Input Image"
#define WINDOW_OUTPUT "Output Image"

using namespace std;
using namespace cv;

Mat img, img_box, templ_Stop, templ_Go;
Mat result;
int match_method = 5;
int max_Trackbar = 5;
Point matchLoc;

int min_score = 50;

bool destroy=false;
CvRect box;
bool drawing_box = false;


int MatchingMethod(Mat templ, Scalar color);
#ifdef DEBUG 
void saveTemplate(CvRect rect, int type);
void my_mouse_callback( int event, int x, int y, int flags, void* param );
#endif

#ifdef DEBUG 
void saveTemplate(CvRect rect, int type)
{
	Mat cropTempl = img(rect);
	switch(type)
	{
		case 0: imwrite( "templ_Stop.jpg", cropTempl ); break;
		case 1: imwrite( "templ_Go.jpg", cropTempl ); break;
	}
}
#endif

#ifdef DEBUG 
// Implement mouse callback
void my_mouse_callback( int event, int x, int y, int flags, void* param )
{
  //Mat* frame = (Mat*) param;

  switch( event )
  {
      case CV_EVENT_MOUSEMOVE: 
      {
          if( drawing_box )
          {
              box.width = x-box.x;
              box.height = y-box.y;
          }
      }
      break;

      case CV_EVENT_LBUTTONDOWN:
      {
          drawing_box = true;
          box = cvRect( x, y, 0, 0 );
      }
      break;

      case CV_EVENT_LBUTTONUP:
      {
          drawing_box = false;
          if( box.width < 0 )
          {
              box.x += box.width;
              box.width *= -1;
          }

          if( box.height < 0 )
          {
              box.y += box.height;
              box.height *= -1;
          }

          //draw_box(frame, box);
      }
      break;

      case CV_EVENT_RBUTTONUP:
      {
          destroy=true;
      }
      break;

      default:
      break;
   }
}
#endif

int main()
{
	VideoCapture cap(2); // open the default camera
    if(!cap.isOpened())  // check if we succeeded
        return -1;

	Size size(320,240);
	#ifdef DEBUG 
		namedWindow( WINDOW_INPUT, CV_WINDOW_AUTOSIZE );
		namedWindow( WINDOW_OUTPUT, CV_WINDOW_AUTOSIZE );
	#endif

	#ifdef DEBUG 
		cvSetMouseCallback(WINDOW_INPUT, my_mouse_callback, NULL);
		int templ_type = 0;
		while(1)
		{
			cap >> img;
			resize(img, img, size);
			img_box = img.clone();
			switch(templ_type)
			{
				case 0: rectangle(img_box, box, Scalar(0, 0, 255), 1, 8, 0); break;
				case 1: rectangle(img_box, box, Scalar(0, 255, 255), 1, 8, 0); break;
			}

			imshow(WINDOW_INPUT, img_box);
		
			char c = waitKey(33);
			switch(c)
			{
				case '1': templ_type = 0; break;
				case '2': templ_type = 1; break;
				case 13: saveTemplate(box, templ_type); break;
			}
			if(c == ' ')
				break;
		}
	#endif

	templ_Stop = imread("templ_Stop.jpg", 1);
	templ_Go = imread("templ_Go.jpg", 1);
	
	while(1)
	{
		static int i = 0;
        cap >> img;
		resize(img, img, size);
		//imshow(WINDOW_INPUT, img);

		#ifdef DEBUG 
			/// Create Trackbar
			char* trackbar_label = "Method: \n 0: SQDIFF \n 1: SQDIFF NORMED \n 2: TM CCORR \n 3: TM CCORR NORMED \n 4: TM COEFF \n 5: TM COEFF NORMED";
			createTrackbar( trackbar_label, WINDOW_INPUT, &match_method, max_Trackbar, NULL );
			createTrackbar( "Score Match", WINDOW_INPUT, &min_score, 100, NULL);
		#endif

		static int Stop_Detect = 0, Go_Detect = 0;
		Stop_Detect = MatchingMethod(templ_Stop, Scalar(0, 0, 255));
		Go_Detect = MatchingMethod(templ_Go, Scalar(0, 255, 255));
		
		#ifdef DEBUG 
			if(Stop_Detect == 1)
				rectangle( img, matchLoc, Point( matchLoc.x + templ_Stop.cols , matchLoc.y + templ_Stop.rows ), Scalar(0, 0, 255), 2, 8, 0 );
			else if(Go_Detect == 1)
				rectangle( img, matchLoc, Point( matchLoc.x + templ_Go.cols , matchLoc.y + templ_Go.rows ), Scalar(0, 255, 255), 2, 8, 0 );
			imshow(WINDOW_INPUT, img);
		#endif

		if(Stop_Detect)
		{
			cout<<"\tStop!!\n";
		}
		else if(Go_Detect)
		{
			cout<<"\tGo!!\n";
		}
		else
		{
			// NONE
		}
		char c = waitKey(33);
		if(c == 27)
			break;
	}
}



int MatchingMethod(Mat templ, Scalar color)
{
	int detect = 0;
  /// Source image to display
  Mat img_display;
  Mat res_temp;
  img.copyTo( img_display );

  /// Create the result matrix
  int result_cols =  img.cols - templ.cols + 1;
  int result_rows = img.rows - templ.rows + 1;

  result.create( result_rows, result_cols, CV_32FC1 );

  /// Do the Matching and Normalize
  matchTemplate( img, templ, result, match_method );
  result.copyTo( res_temp );
  normalize( result, result, 0, 1, NORM_MINMAX, -1, Mat() );

  /// Localizing the best match with minMaxLoc
  double minVal; double maxVal; Point minLoc; Point maxLoc;
  //Point matchLoc;

  minMaxLoc( result, &minVal, &maxVal, &minLoc, &maxLoc, Mat() );

  /// For SQDIFF and SQDIFF_NORMED, the best matches are lower values. For all the other methods, the higher the better
  if( match_method  == CV_TM_SQDIFF || match_method == CV_TM_SQDIFF_NORMED )
    { matchLoc = minLoc; }
  else
    { matchLoc = maxLoc; }

	double minVal_score; double maxVal_score;
	minMaxIdx( res_temp, &minVal_score, &maxVal_score);
	minVal_score = minVal_score * 100;
	maxVal_score = maxVal_score * 100;
	//cout <<minVal_score <<'\t'<<maxVal_score <<'\n';
	if( maxVal_score > min_score )
	{
		/// Show me what you got
		//rectangle( img_display, matchLoc, Point( matchLoc.x + templ.cols , matchLoc.y + templ.rows ), color, 2, 8, 0 );
		//rectangle( result, matchLoc, Point( matchLoc.x + templ.cols , matchLoc.y + templ.rows ), Scalar(0, 0, 255), 2, 8, 0 );

		detect = 1;
	}

	#ifdef DEBUG 
		//imshow( WINDOW_INPUT, img_display );
		imshow( WINDOW_OUTPUT, result );
	#endif

  return detect;
}

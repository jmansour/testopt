#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;
using namespace std;

#include "SimpleFlowDenseOF.h"

int main( int argc, char** argv )
{
   unsigned numImages = argc - 1;
   if( numImages < 2 ){
      cout << "You must provide at least two images! Fool!" << endl;
      exit(1);
   }

   vector<Mat> inputImages;
   for (unsigned ii=0; ii<numImages; ii++)
      inputImages.push_back( imread(string(argv[ii+1])) );

   SimpleFlowDenseOF simpleFlow;
   simpleFlow.setImages( inputImages );
   simpleFlow.setOmegaRadius( 3 );
   simpleFlow.setetaRadius( 3 );
   simpleFlow.setPyramidLevels( 2 );
   
   namedWindow( "we do test", CV_WINDOW_AUTOSIZE );
   Mat resultImage(inputImages[0].rows, inputImages[0].cols, CV_32FC2);
//   imshow("we do test", resultImage);
//   waitKey();
   simpleFlow.calcFlow(resultImage);
   return 0;
}
   

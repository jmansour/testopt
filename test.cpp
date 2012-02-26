#include <iostream>

#include "cv.h"
#include "highgui.h"

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
   char* inputImageNames[argc-1];
   for (int ii=0; ii<argc-1; ii++)
      inputImageNames[ii] = argv[ii+1];

   vector<Mat> inputImages;
   for (int ii=0; ii<argc-1; ii++)
      inputImages.push_back( imread(inputImageNames[ii]) );

   SimpleFlowDenseOF testEmptyOF;
   testEmptyOF.setImages( inputImages );

   vector<Mat> &inputImages2 = testEmptyOF.getImages();
   
   cout << "the number of images is " << testEmptyOF.getNumImages() << endl;
   cout << "the number of images is (var2) " << inputImages2.size() << endl;
   return 0;
}
   

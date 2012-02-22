#include <iostream>

#include "cv.h"
#include "highgui.h"

using namespace cv;
using namespace std;

#include "DenseOF.h"

int main( int argc, char** argv )
{
   if( argc < 3 ){
      cout << "You must provide at least two images! Fool!" << endl;
      exit;
   }
   char* inputImageNames[argc-1];
   for (int ii=0; ii<argc-1; ii++)
      inputImageNames[ii] = argv[ii+1];

   vector<Mat> inputImages;
   for (int ii=0; ii<argc-1; ii++)
      inputImages.push_back( imread(inputImageNames[ii]) );

   DenseOF testEmptyOF;
   testEmptyOF.setImages( inputImages );

   vector<Mat> &inputImages2 = testEmptyOF.getImages();
   
   cout << "the number of images is " << testEmptyOF.getNumImages() << endl;
   cout << "the number of images is (var2) " << inputImages2.size() << endl;
   return 0;
}
   

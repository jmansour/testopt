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

   vector<Mat> inputImages;
   for (unsigned ii=0; ii<numImages; ii++)
      inputImages.push_back( imread(string(argv[ii])) );

   SimpleFlowDenseOF simpleFlow;;
   simpleFlow.setImages( inputImages );

   return 0;
}
   

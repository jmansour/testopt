#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

#include "SimpleFlowDenseOF.h"


// POTENTIAL OPTIMISATIONS:
// * move calculation of offets elsewhere as it only needs to be calculated once
// * check to see if handling out of bounds for norms can be done fasters
// * check to see if norms can handled better then using pow... perhaps images of shorts is better than ints

bool SimpleFlowDenseOF::calcFlow( Mat &resultImage ){
   assert(numImages == 2 && (resultImage.type() == CV_32FC2));
   
   // first build image pyramids
   vector<Mat> image0Pyramid;
   vector<Mat> image1Pyramid;
   buildPyramid( (*images)[0], image0Pyramid, numPyramidLevels );
   buildPyramid( (*images)[1], image1Pyramid, numPyramidLevels );

   // great a currentflow Mat, and init to zero
   Mat currentFlow = Mat::zeros(image0Pyramid[numPyramidLevels].rows, image0Pyramid[numPyramidLevels].cols, CV_32SC2);
   // run flow
   for (int ii = numPyramidLevels; ii>0; ii--) {
      calcFlowAtLevel( image0Pyramid[ii], image1Pyramid[ii], currentFlow, resultImage );
      pyrUp( resultImage,  currentFlow, Size(resultImage.cols*2, (resultImage.rows*2) ) );
      currentFlow = 2*currentFlow;  // multiply by two to account for image scaling
   }
   
   // compute final full resolution flow
   calcFlowAtLevel( image0Pyramid[0], image1Pyramid[0], currentFlow, resultImage );

   return 1;
}

bool SimpleFlowDenseOF::calcFlowAtLevel( Mat &fromImage, Mat &toImage, Mat &initFlow, Mat &levelResult ){
   int successFlag;
   //// first calculate the rgb norms
   // setup required mats
   Mat norms(fromImage.rows, fromImage.cols, CV_32FC(15));
   // go ahead a calculate means
   successFlag = calcNormsWithMean( fromImage, toImage, initFlow, norms ); assert(successFlag);
   
   // create energy term which is the normed difference of the two images
   
   return 1;
}

bool SimpleFlowDenseOF::calcNormsWithMean( Mat &fromImage, Mat &toImage, Mat &initFlow, Mat &norms ){
   assert(  fromImage.isContinuous() && toImage.isContinuous() && norms.isContinuous() );
   assert(  fromImage.rows == toImage.rows && 
            fromImage.cols == toImage.cols &&
            fromImage.elemSize() == toImage.elemSize() &&
            fromImage.rows == initFlow.rows &&
            fromImage.cols == initFlow.cols  );

   //// build offset vec (prob should do somewhere else)
   int OmegaRadiusInt = (int) OmegaRadius;
   int OmegaRadiusInt2 = OmegaRadiusInt*OmegaRadiusInt;
   int offsets1D[OmegaRadiusInt2];
   int vecy[OmegaRadiusInt2], vecx[OmegaRadiusInt2];
   
   unsigned posCount = 0;
   for (int ii=-OmegaRadiusInt; ii<=OmegaRadiusInt; ii++) {
      for (int jj=-OmegaRadiusInt; jj<=OmegaRadiusInt; jj++) {
         offsets1D[posCount] = ii*toImage.step + jj*toImage.elemSize();
         vecy[posCount] = ii;
         vecx[posCount] = jj;
         posCount++;
      }
   }
   //// go through all pixels of fromImage, determine norms
   for (int ii=0; ii<fromImage.rows; ii++) {
      for (int jj=0; jj<fromImage.cols; jj++) {
         // get fromImage pixel
         const uchar* fImgPix = fromImage.ptr<uchar>(ii) + jj*fromImage.elemSize();
         float* normPix = norms.ptr<float>(ii) + jj*norms.elemSize();
         // get initFlow displacement
         const int*  initFlowPix = initFlow.ptr<int>(ii) + jj*initFlow.elemSize();
         // get centre of Omega
         int iiOm = ii + initFlowPix[0];
         int jjOm = jj + initFlowPix[1];
         uchar* tImgCentrePix = toImage.ptr<uchar>(iiOm) + jjOm*toImage.elemSize();
         // first check if test support may fall outside toImage, if so do the following:
         if( iiOm > OmegaRadiusInt && iiOm < toImage.rows-OmegaRadiusInt &&
             jjOm > OmegaRadiusInt && jjOm < toImage.cols-OmegaRadiusInt)
         {       // no bounds checks required
            for (int aa=0; aa<OmegaRadiusInt2; aa++) {
               *normPix = 0;
               uchar* tImgPix = tImgCentrePix + offsets1D[aa];
               for (int bb=0; bb<fromImage.channels(); bb++) {
                  *normPix += pow((float)(*(fImgPix+bb) - *(tImgPix+bb)), 2);
               }
               normPix++;
            }
         } else {  // bounds checks required here
            for (int aa=0; aa<OmegaRadiusInt2; aa++) {
               int iiPix = iiOm + vecy[aa];
               if (iiPix < 0 || iiPix >= toImage.rows){    // if outside image vertically, set to FLT_MAX
                  *normPix = FLT_MAX;
               } else {
                  int jjPix = jjOm + vecx[aa];
                  if (jjPix < 0 || jjPix >= toImage.cols){ // if outside image horizontally, set to FLT_MAX
                     *normPix = FLT_MAX;
                  } else {                                 // else calc norm
                     *normPix = 0;
                     uchar* tImgPix = tImgCentrePix + offsets1D[aa];
                     for (int bb=0; bb<fromImage.channels(); bb++) {
                        *normPix += pow((float)(*(fImgPix+bb) - *(tImgPix+bb)), 2);
                     }
                  }
               }
               normPix++;
            }
            
         }
      }
   }
   return 1;
}















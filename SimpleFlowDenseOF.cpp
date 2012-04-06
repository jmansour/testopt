#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

#include "SimpleFlowDenseOF.h"


// POTENTIAL OPTIMISATIONS:
// * move calculation of offets elsewhere as it only needs to be calculated once.
// * check to see if handling out of bounds for norms can be done fasters.
// * check to see if norms can handled better then using pow... perhaps images of shorts is better than ints.
// * table for sigma_c calculations.. with and without interpolation.

bool SimpleFlowDenseOF::calcFlow( Mat &resultImage ){
   assert(numImages == 2 && (resultImage.type() == CV_32FC2));
   
   // first build image pyramids
   vector<Mat> image0Pyramid;
   vector<Mat> image1Pyramid;
   buildPyramid( (*images)[0], image0Pyramid, numPyramidLevels );
   buildPyramid( (*images)[1], image1Pyramid, numPyramidLevels );

   Mat currentFlow[numPyramidLevels+1];
   Mat     newFlow[numPyramidLevels+1];
   // create Mats for the current level solution
   for (unsigned ii = 0; ii<=numPyramidLevels; ii++) {
      currentFlow[ii] = Mat( image0Pyramid[numPyramidLevels].rows, image0Pyramid[numPyramidLevels].cols, CV_32SC2 );
          newFlow[ii] = Mat( image0Pyramid[numPyramidLevels].rows, image0Pyramid[numPyramidLevels].cols, CV_32SC2 );
   }
   // run flow at top (smallest) pyramid level 
   currentFlow[numPyramidLevels] = Mat::zeros(currentFlow[numPyramidLevels].rows, currentFlow[numPyramidLevels].cols, CV_32SC2 );   // init top to zero
   calcFlowAtLevel( image0Pyramid[numPyramidLevels], image1Pyramid[numPyramidLevels], currentFlow[numPyramidLevels], newFlow[numPyramidLevels] );
   for (int ii = numPyramidLevels-1; ii>=0; ii--) {
      newFlow[ii+1] = 2*newFlow[ii+1];    // multiply by two to prepare for new image scaling
      pyrUp( currentFlow[ii+1], currentFlow[ii], Size(currentFlow[ii+1].cols*2, currentFlow[ii+1].rows*2 ) );
      calcFlowAtLevel( image0Pyramid[ii], image1Pyramid[ii], currentFlow[ii], newFlow[ii] );
   }
   
   // compute final full resolution flow
   //calcFlowAtLevel( image0Pyramid[0], image1Pyramid[0], currentFlow, resultImage );

   return 1;
}

bool SimpleFlowDenseOF::calcFlowAtLevel( Mat &fromImage, Mat &toImage, Mat &currentFlow, Mat &newFlow ){
   bool successFlag;
   assert( fromImage.rows == toImage.rows ); 
   assert( fromImage.cols == toImage.cols );
   assert( fromImage.rows == currentFlow.rows );
   assert( fromImage.cols == currentFlow.cols );
   assert( fromImage.rows == newFlow.rows );
   assert( fromImage.cols == newFlow.cols );
   assert( fromImage.elemSize() == toImage.elemSize() );

   //// first calculate the rgb norms
   // setup required mats
   Mat norms(fromImage.rows, fromImage.cols, CV_32FC(OmegaPixelCount));
   // go ahead a calculate means
   successFlag = calcNormsWithMean( fromImage, toImage, currentFlow, norms ); assert(successFlag);
//   vector<Mat> splitImages;
//   split(norms, splitImages);
//   for (unsigned kk=0; kk<OmegaPixelCount; kk++) {
//      imshow( "norm images", splitImages[kk]);
//      waitKey();
//   }
   
   successFlag = filterThenFindEnergyMinimiser( fromImage, norms, currentFlow, newFlow ); assert(successFlag);
   
   return successFlag;
}

bool SimpleFlowDenseOF::calcNormsWithMean( Mat &fromImage, Mat &toImage, Mat &currentFlow, Mat &norms ){
   assert( fromImage.rows == norms.rows ); 
   assert( fromImage.cols == norms.cols );
   assert( fromImage.isContinuous() && toImage.isContinuous() && norms.isContinuous() );
   
   // build offset vec (prob should do somewhere else)
   int OmegaRadiusInt = (int) OmegaRadius;
   int offsets1D[OmegaPixelCount];
   int vecy[OmegaPixelCount], vecx[OmegaPixelCount];
   
   unsigned posCount = 0;
   for (int ii=-OmegaRadiusInt; ii<=OmegaRadiusInt; ii++) {
      for (int jj=-OmegaRadiusInt; jj<=OmegaRadiusInt; jj++) {
         offsets1D[posCount] = ii*toImage.step + jj*toImage.channels();
         vecy[posCount] = ii;
         vecx[posCount] = jj;
         posCount++;
      }
   }
   assert(posCount == OmegaPixelCount);
   // go through all pixels of fromImage, determine norms
   for (int ii=0; ii<fromImage.rows; ii++) {
      
      const uchar*      fImgRowPtr = fromImage.ptr<uchar>(ii);
      float*            normRowPtr =     norms.ptr<float>(ii);
      const int* currentFlowRowPtr = currentFlow.ptr<int>(ii);
 
      for (int jj=0; jj<fromImage.cols; jj++) {
         
         const uchar*      fImgPix =        fImgRowPtr +   fromImage.channels()*jj;          // get fromImage pixel
         float*            normPix =        normRowPtr +       norms.channels()*jj;          // get norm pixel
         const int* currentFlowPix = currentFlowRowPtr + currentFlow.channels()*jj;          // get currentFlow displacement

         int iiOm = ii + currentFlowPix[0];                                                  // get centre of Omega
         int jjOm = jj + currentFlowPix[1];
         uchar* tImgCentrePix = toImage.ptr<uchar>(iiOm) + jjOm*toImage.channels();          // get pixel at centre of Omega

         if( (iiOm > OmegaRadiusInt) && (iiOm < toImage.rows-OmegaRadiusInt) &&              // check if test support may fall outside toImage
             (jjOm > OmegaRadiusInt) && (jjOm < toImage.cols-OmegaRadiusInt) )
         {                                                                                   // no bounds checks required here
            for (unsigned aa=0; aa<OmegaPixelCount; aa++) {
               *normPix = 0;
               uchar* tImgPix = tImgCentrePix + offsets1D[aa];
               for (int bb=0; bb<fromImage.channels(); bb++) {
                  *normPix += pow((float)(*(fImgPix+bb) - *(tImgPix+bb)), 2);
               }
               normPix++;
            }
         } else {                                                                            // bounds checks required here
            for (unsigned aa=0; aa<OmegaPixelCount; aa++) {
               int iiPix = iiOm + vecy[aa];
               if (iiPix < 0 || iiPix >= toImage.rows){                                      // if outside image vertically, set to FLT_MAX
                  *normPix = FLT_MAX;
               } else {
                  int jjPix = jjOm + vecx[aa];
                  if (jjPix < 0 || jjPix >= toImage.cols){                                   // if outside image horizontally, set to FLT_MAX
                     *normPix = FLT_MAX;
                  } else {                                                                   // else calc norm
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


bool SimpleFlowDenseOF::filterThenFindEnergyMinimiser( Mat &fromImage, Mat &norms, Mat &currentFlow, Mat &newFlow ){
   assert( fromImage.rows == norms.rows );
   assert( fromImage.cols == norms.cols );
   assert( fromImage.rows == currentFlow.rows );
   assert( fromImage.cols == currentFlow.cols );
   assert( fromImage.rows == newFlow.rows );
   assert( fromImage.cols == newFlow.cols );
   assert( currentFlow.elemSize() == newFlow.elemSize() );
   
   // calculate Mat containing pixel displacement factor
   Mat dispFact(2*etaRadius+1,    2*etaRadius+1, CV_32FC1);
   Mat dispOffsets(2*etaRadius+1, 2*etaRadius+1, CV_32SC1);
   // get pointer to image centre pixel
   assert(0) // double check the .step result
   float*        dispCentrePix =    (float*)dispFact.data + etaRadius*dispFact.step    + etaRadius;
   int*   dispOffsetsCentrePix =   (int*)dispOffsets.data + etaRadius*dispOffsets.step + etaRadius;
   for (int ii = -etaRadius; ii<=etaRadius; ii++) {
      for (int jj = -etaRadius; jj<=etaRadius; jj++) {
         ii*dispFact.step    + jj + dispCentrePix        = exp(-pow((ii*ii + jj*jj),2)/(2*sigma_d));
         ii*dispOffsets.step + jj + dispOffsetsCentrePix = ii*fromImage.step + jj*fromImage.channels();
      }
   }
   
   // go through all pixels of levelResult, set pixel to direction which minimises energy
   for (int ii=0; ii<newFlow.rows; ii++) {

      const uchar*      fImgRowPtr = fromImage.ptr<uchar>(ii);
      const float*     normsRowPtr = norms.ptr<float>(ii);
      const int*     newFlowRowPtr = newFlow.ptr<int>(ii);
      const int* currentFlowRowPtr = currentFlow.ptr<int>(ii);

      for (int jj=0; jj<newFlow.cols; jj++) {

         const uchar*  fImgPix     = fImgRowPtr        + jj*fromImage.channels();                          // get fromImage pixel
         const float* normsPix     = normsRowPtr       + jj*norms.channels();                              // get norms pixel
         const int* currentFlowPix = currentFlowRowPtr + jj*currentFlow.channels();                        // get currentFlow pixel
         const int* newFlowPix     = newFlowRowPtr     + jj*newFlow.channels();                            // get newFlow pixel
         
         // perform a bilateral filter for each candidate vector over neighbouring pixels to (ii,jj), ie all pixels in Eta
         // first calculate colour different factor for all pixels in eta
         for (int ll=-etaRadius; ll<=etaRadius; ll++) {
            for(int mm=-etaRadius; mm<=etaRadius; mm++) {
               
            }
         }
         // now, need to go through each candidate vector in Omega, and assign an energy (perhaps should normalise for vecors with less contributions (at boundaries))
         
         
      }
   }
   return 1;
   
}



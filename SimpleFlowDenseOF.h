#pragma once

#include "DenseOF.h"

class SimpleFlowDenseOF: public DenseOF
{
public:
   SimpleFlowDenseOF()                              { setOmegaRadius(0); setetaRadius(0); } ;
   ~SimpleFlowDenseOF()                             {};
   bool calcFlow( Mat &resultImage );
   void setOmegaRadius( const unsigned OmRadius ) { OmegaRadius = OmRadius; OmegaPixelCount = (2*OmRadius+1)*(2*OmRadius+1);};
   void setetaRadius( const unsigned etRadius ) { etaRadius = etRadius; etaPixelCount = (2*etRadius+1)*(2*etRadius+1); };
   void setsigma_d( const float sigma_d_in ) { sigma_d = sigma_d_in; assert(sigma_d > 0); };
   void setsigma_c( const float sigma_c_in ) { sigma_c = sigma_c_in; assert(sigma_c > 0); };
private:
   bool calcFlowAtLevel( Mat &fromImage, Mat &toImage, Mat &currentFlow, Mat &newFlow );
   bool calcNormsWithMean( Mat &fromImage, Mat &toImage, Mat &currentFlow, Mat &norms );
   bool filterThenFindEnergyMinimiser( Mat &fromImage, Mat &norms, Mat &currentFlow, Mat &newFlow );
   void createMapping();
   unsigned OmegaRadius;
   unsigned OmegaPixelCount;
   unsigned etaRadius;
   unsigned etaPixelCount;
   float sigma_d;
   float sigma_c;
   Mat mapto1DOmega;
   vector<int> vecxOmega;
   vector<int> vecyOmega;
   Mat mapto1Deta;
   vector<int> vecxeta;
   vector<int> vecyeta;
};

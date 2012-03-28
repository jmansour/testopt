#pragma once

#include "DenseOF.h"

class SimpleFlowDenseOF: public DenseOF
{
public:
   SimpleFlowDenseOF()                              { setOmegaRadius(0); setetaRadius(0); } ;
   ~SimpleFlowDenseOF()                             {};
   bool calcFlow( Mat &resultImage );
   void setOmegaRadius( const unsigned OmRadius ) { OmegaRadius = OmRadius; };
   void setetaRadius( const unsigned etRadius ) { etaRadius = etRadius; };
private:
   bool calcFlowAtLevel( Mat &fromImage, Mat &toImage, Mat &initFlow, Mat &levelResult );
   bool calcNormsWithMean( Mat &fromImage, Mat &toImage, Mat &initFlow, Mat &norms );
   unsigned OmegaRadius;
   unsigned etaRadius;
};

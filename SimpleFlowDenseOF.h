#pragma once

#include "DenseOF.h"

class SimpleFlowDenseOF: public DenseOF
{
public:
   SimpleFlowDenseOF()                              {};
   ~SimpleFlowDenseOF()                             {};
   bool calcFlow( Mat &resultImage, Mat *resultImageScale ) ;
};

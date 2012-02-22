

class DenseOF
{
public:
   DenseOF()                                        { images = NULL; numImages = 0; numPyramidLevels = 0; };
   ~DenseOF()                                       { releaseImages(); };
   void setImages( vector<Mat> & flowImages )       { images = &flowImages; numImages = images->size(); };
   vector<Mat> & getImages() const                  { return *images; };
   int  getNumImages() const                        { return numImages; };
   void releaseImages()                             { images = NULL; };
   void setPyramidLevels( const int levels )        { numPyramidLevels = levels; };
   int  getPyramidLevels() const                    { return numPyramidLevels; };
   //virtual bool calcFlow( Mat &resultImage, Mat *resultImageScale ) =0;
protected:
   vector<Mat> * images;
   int numImages;                // reference to set of images
   unsigned numPyramidLevels;   
};


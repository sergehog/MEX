//////////////////////////////////////////////////////////////////////////
//
// exrread.cpp
//
// Matlab interface for reading a float image to exr file
//
// img = exrread(filename)
//
// img is a 3d (color) hdr image (when filename is gray image, img's 3 
// channels are the same).
//
// see also exrwrite.cpp
// 
// Jinwei Gu. 2006/02/10
// jwgu@cs.columbia.edu
//
// When using mex to compiling it in matlab, make sure to use VC7.1 or above
// instead of VC6. 
//////////////////////////////////////////////////////////////////////////
#ifdef __cplusplus 
extern "C" {
#include "mex.h"
}
#endif // __cplusplus

#include "math.h"
#include "stdio.h"

#include "IlmImf/ImathBox.h"
#include "IlmImf/ImfRgba.h"
#include "IlmImf/ImfRgbaFile.h"

//#pragma comment(lib, "IlmImf_dll.lib")
//#pragma comment(lib, "Half.lib")

#pragma comment (lib, "libeng.lib") 
#pragma comment (lib, "libmx.lib")
#pragma comment (lib, "libmex.lib")

//#pragma comment(lib, "D:/OpenEXR-1.2.2-win32/lib/IlmImf_dll.lib")
//#pragma comment(lib, "D:/OpenEXR-1.2.2-win32/lib/Half.lib")

using namespace Imf;
using namespace Imath;

void mexFunction (int nout, mxArray* output[], int in, const mxArray* input[])
//void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) 
{ 
    if(nout != 1 && in != 1)
    {
        mexErrMsgTxt("exrread: usage: I = exrread('filename');\n\r");
    }
    char inputfile[256];
    mxGetString(input[0], inputfile, 256);
    RgbaInputFile file(inputfile);
	Box2i dw = file.dataWindow();

	int width  = dw.max.x - dw.min.x + 1;
	int height = dw.max.y - dw.min.y + 1;

	struct Rgba *pixels = new struct Rgba[width*height];

	file.setFrameBuffer(pixels-dw.min.x-dw.min.y*width, 1, width);
	file.readPixels(dw.min.y, dw.max.y);
	
	int i, j, k, A, m; 
	size_t dims[3] = {height, width, 3};
	//dims[0]=height; dims[1]=width; dims[2]=3;
	output[0] = mxCreateNumericArray(3, dims, mxDOUBLE_CLASS, mxREAL); 
	double *img = mxGetPr(output[0]);
	A = width*height;
	
	for(i=0; i<height; i++)
	for(j=0; j<width; j++)
	{
		k = i*width+j;
		m = j*height+i;

		img[m    ] = pixels[k].r;
		img[A+m  ] = pixels[k].g;
		img[2*A+m] = pixels[k].b;
	}
	delete[] pixels; pixels = NULL;
} 

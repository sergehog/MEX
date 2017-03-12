/** 
*	@file render_opengl.cpp
*	@date 24.07.2012
*	@author Sergey Smirnov <sergey.smirnov@tut.fi>
*/

#include <GL/glew.h>
#include <GL/freeglut.h>

//#ifdef __cplusplus 
extern "C" {
#include "mex.h"
}
//#endif // __cplusplus

#include "../common/meximage.h"
//#include "../common/defines.h"
//#include "../common/matching.h"

#include <cmath>
#include <algorithm>
//#include <fstream>
//#include <iostream>
//#include <sstream>
//#include <omp.h>

#pragma comment (lib, "libeng.lib") 
#pragma comment (lib, "libmx.lib")
#pragma comment (lib, "libmex.lib")

//#pragma comment (lib, "freeglut_static.lib")

//typedef unsigned char uint8;
using namespace mymex;


void mexFunction (int nout, mxArray* output[], int in, const mxArray* input[])
{	
	//omp_set_num_threads(std::max(2,omp_get_num_threads()));
	//omp_set_dynamic(0);

	if(nout != 1 || in != 3 || mxGetClassID(input[0])!=mxSINGLE_CLASS || mxGetClassID(input[1])!=mxSINGLE_CLASS || mxGetClassID(input[2])!=mxSINGLE_CLASS)
	{
		mexErrMsgTxt("USAGE: Output = render_opengl(single(Image), single(Z), single(P_camera));\n"); 
    } 

	MexImage<float> Image(input[0]);
	MexImage<float> Z(input[1]);
	MexImage<float> P(input[2]);

	const int width = Image.width;
	const int height = Image.height;
	const int colors = Image.layers;
	const long HW = width*height;

	if(Z.width != width || Z.height != height)
	{
		mexErrMsgTxt("ERROR: width and height of 'Image' and 'Z' images must coincide!\n");
	}
	
	//const int layers = Input.layers;
	//const int radius  = std::max(0, (int)mxGetScalar(input[2]));
	//float const sigma_color = (in > 3) ? std::max(0.1f, (float)mxGetScalar(input[3])) : 30.f;	
	//const int range_max = 255;
	//const int PBFICs = (in > 4) ? std::min(range_max, std::max(4, (int)mxGetScalar(input[4]))) : 4;	
	//const float PBFIC_step = (float)range_max/(PBFICs-1);
	
	size_t dimms[] = {height, width, colors};
	output[0] = mxCreateNumericArray(3, dimms, mxSINGLE_CLASS, mxREAL);
	MexImage<float> Output(output[0]);
	int zero = 0;
	//glutInit(&zero, NULL);

	//wglCreateContext(hDC)
	//GLuint program = glCreateProgram();




}

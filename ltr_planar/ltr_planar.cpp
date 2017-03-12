/** Extends classical Left-to-Right check to planar-disparity-model
* @file ltr_planar.cpp
* @date 16.05.2016
* @author Sergey Smirnov
*/

#ifdef __cplusplus 
extern "C" {
#include "mex.h"
}
#endif // __cplusplus

#include <cstdint>
#include "../common/meximage.h"
//#include <cmath>

#pragma comment (lib, "libeng.lib") 
#pragma comment (lib, "libmx.lib")
#pragma comment (lib, "libmex.lib")

using namespace mymex;

template<int direction>
void check_validity(const MexImage<const float> &Aref, const MexImage<const float> &Atmpl, const MexImage<bool> &Valid, const float thr_d, const float thr_ab)
{
	const int width = Aref.width;
	const int height = Aref.height;
	const int64_t HW = static_cast<int64_t>(Aref.layer_size);

#pragma ompr parallel for
	for (int64_t i = 0; i < HW; i++)
	{
		const int x = i / height;
		const int y = i % height;
		const float a = Aref(x, y, 0);
		const float b = Aref(x, y, 1);
		const float c = Aref(x, y, 2);
		const float d = x*a + y*b + c;		
		const int dl = static_cast<int>(floor(d));
		const int du = static_cast<int>(ceil(d));
		const int xl = (x + direction*dl);
		const int xu = (x + direction*du);

		if (xl >= 0 && xl < width)
		{
			const float at = Atmpl(xl, y, 0);
			const float bt = Atmpl(xl, y, 1);
			const float ct = Atmpl(xl, y, 2);
			const float dt = at*xl + bt*y + ct;
			if (abs(dt - d) <= thr_d && abs(at - a) <= thr_ab && abs(bt - b) <= thr_ab)
			{
				Valid(x, y) = true;
			}
		}

		if (xu >= 0 && xu < width)
		{
			const float at = Atmpl(xu, y, 0);
			const float bt = Atmpl(xu, y, 1);
			const float ct = Atmpl(xu, y, 2);
			const float dt = at*xu + bt*y + ct;
			if (abs(dt - d) <= thr_d && abs(at - a) <= thr_ab && abs(bt - b) <= thr_ab)
			{
				Valid(x, y) = true;
			}
		}
	}
}

void mexFunction(int nout, mxArray* output[], int in, const mxArray* input[])
{
	if (in != 4 || nout != 2 || mxGetClassID(input[0]) != mxSINGLE_CLASS || mxGetClassID(input[1]) != mxSINGLE_CLASS)
	{
		mexErrMsgTxt("Left-to-Right Consistency Check \n USAGE: [ValidL, ValidR] = ltr_check(AL, AR, thr_d, thr_ab);");
	}
	const MexImage<const float> Al(input[0]);
	const MexImage<const float> Ar(input[1]);
	const float thr_d = static_cast<float>(mxGetScalar(input[2]));
	const float thr_ab = static_cast<float>(mxGetScalar(input[3]));	

	const int width = Al.width;
	const int height = Al.height;	
	const int64_t HW = Al.layer_size;
	if (height != Ar.height || width != Ar.width)
	{
		mexErrMsgTxt("ERROR: Sizes of 'Al' and 'Ar' must be the same.");
	}

	const mwSize dims[] = {height, width, 1};
	output[0] = mxCreateNumericArray(3, dims, mxLOGICAL_CLASS, mxREAL);
	output[1] = mxCreateNumericArray(3, dims, mxLOGICAL_CLASS, mxREAL);

	const MexImage<bool> ValidL(output[0]);
	const MexImage<bool> ValidR(output[1]);

#pragma omp parallel sections
	{
#pragma omp section
		{
			ValidL.setval(0);
			check_validity<-1>(Al, Ar, ValidL, thr_d, thr_ab);
		}
#pragma omp section
		{
			ValidR.setval(0);
			check_validity<+1>(Ar, Al, ValidR, thr_d, thr_ab);
		}
	}
	
}



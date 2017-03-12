/** Computing a cost volume for a stereo-pair, using tricks suggested by Q. Yang
*	@file compute_cost_yang.cpp
*	@date 31.01.2014
*	@author Sergey Smirnov <sergey.smirnov@tut.fi>
*/

#ifdef __cplusplus 
extern "C" {
#include "mex.h"
}
#endif // __cplusplus

#include "../common/meximage.h"
//#include "../common/defines.h"
#include <vector>
#include <cmath>
#include <algorithm>
#ifndef _DEBUG
#include <omp.h>
#endif

#pragma comment (lib, "libeng.lib") 
#pragma comment (lib, "libmx.lib")
#pragma comment (lib, "libmex.lib")

//typedef unsigned char uint8;
using namespace mymex;

template<typename T>
inline double rgb2gray(MexImage<T> &Image, const int x, const int y)
{
	if (Image.layers == 3)
	{
		return Image(x, y, 0)*0.299 + Image(x, y, 1)*0.587 + Image(x, y, 2)*0.114;
	}
	else
	{
		double avg = 0;
		for (int c = 0; c < Image.layers; c++)
		{
			avg += T(Image(x, y, c));
		}
		return avg / Image.layers;
	}
}


template<typename T>
void compute_gradient(MexImage<T> &Image, MexImage<T> &Gradient, const double halfmax)
{
	const int height = Image.height;
	const int width = Image.width;
	const int colors = Image.layers;
	#pragma omp parallel for
	for (int y = 0; y < height; y++)
	{
		double prev = rgb2gray(Image, 0, y);
		double next = rgb2gray(Image, 1, y);
		double curr = next;
		Gradient(0, y) = T(next - prev + halfmax);
		for (int x = 1; x < width - 1; x++)
		{
			next = rgb2gray(Image, 1, y);
			Gradient(1, y) = T(0.5*(next - prev) + halfmax);
			prev = curr;
			curr = next;
		}
	}
}

//const double max_color_difference = 7;
//const double max_gradient_difference = 2;
//const double weight_on_color = 0.11;

const double max_color_difference = 10;
const double max_gradient_difference = 5;
const double weight_on_color = 0.5;
template<typename T>
void compute_cost(MexImage<T> &Reference, MexImage<T> &Target, MexImage<T> &ReferenceGradient, MexImage<T> &TargetGradient, MexImage<T> &Cost, const int mindisp, const int maxdisp, const double cost_thr)
{
	const int height = Reference.height;
	const int width = Reference.width;
	const int colors = Reference.layers;
	const int layers = Cost.layers;
	const long HW = height*width;
	const float nan = sqrt(-1.f);
	
	const bool left = maxdisp > mindisp;
	
	for (int d = 0; d < layers; d++)
	{
		const int disp = mindisp + d * (left ?  1 : -1);
		#pragma omp parallel for
		for (long i = 0; i < HW; i++)
		{
			const int x = i / height;
			const int y = i % height;
			const int tx = (x - disp) >= width ? width-1 : (x - disp) < 0 ? 0 : (x - disp);

			double cost = 0;
			for (int c = 0; c < colors; c++)
			{ 
				cost += std::abs(double(Reference(x, y, c)) - double(Target(tx, y, c)));
			}
			cost = std::min(cost / 3, max_color_difference);
			double cost_gradient = std::abs(double(ReferenceGradient(x, y)) - double(TargetGradient(tx, y)));
			cost_gradient = std::min(cost_gradient, max_gradient_difference);
			Cost(x, y, d) = T((weight_on_color*cost + (1 - weight_on_color)*cost_gradient) * 255 / (max_color_difference*weight_on_color + (1 - weight_on_color)*max_gradient_difference));
		}
	}
}

void mexFunction(int nout, mxArray* output[], int in, const mxArray* input[])
{
#ifndef _DEBUG
	omp_set_num_threads(std::max<int>(4, omp_get_max_threads()));
	omp_set_dynamic(std::max<int>(2, omp_get_max_threads() / 2));
#endif

	if (in < 4 || in > 5 || nout < 1 || nout > 2)
	{
		mexErrMsgTxt("USAGE: [CostL, <CostR>] = compute_cost_yang(Left, Right, mindisp, maxdisp <, cost_thr>);");
	}

	const int mindisp = (int)mxGetScalar(input[2]);
	const int maxdisp = (int)mxGetScalar(input[3]);
	const double cost_thr = in > 4 ? mxGetScalar(input[4]) : 40.0;

	if (!(maxdisp > mindisp))
	{
		mexErrMsgTxt("maxdisp must be larger than mindisp!");
	}
	
	const int layers = maxdisp - mindisp + 1;
	
	if (mxGetClassID(input[0]) == mxUINT8_CLASS && mxGetClassID(input[1]) == mxUINT8_CLASS)
	{
		MexImage<unsigned char> Left(input[0]);
		MexImage<unsigned char> Right(input[1]);
		if (Left.width != Right.width || Left.height != Right.height || Left.layers != Right.layers)
		{
			mexErrMsgTxt("Resolution of Left and Right images must coincide!!");
		}

		const size_t dims[] = { (size_t)Left.height, (size_t)Left.width, (size_t)layers };
		output[0] = mxCreateNumericArray(3, dims, mxUINT8_CLASS , mxREAL);
		
		MexImage<unsigned char> CostL(output[0]);
		MexImage<unsigned char> GradientL(Left.width, Left.height);
		MexImage<unsigned char> GradientR(Left.width, Left.height);
		#pragma omp parallel sections
		{
			#pragma omp section
			{
				compute_gradient<unsigned char>(Left, GradientL, 127.5);
			}
			#pragma omp section
			{
				compute_gradient<unsigned char>(Right, GradientR, 127.5);
			}
		}		
		
		compute_cost<unsigned char>(Left, Right, GradientL, GradientR, CostL, mindisp, maxdisp, cost_thr);
		if (nout > 1)
		{
			output[1] = mxCreateNumericArray(3, dims, mxUINT8_CLASS, mxREAL);
			MexImage<unsigned char> CostR(output[1]);
			compute_cost<unsigned char>(Right, Left, GradientR, GradientL, CostR, -mindisp, -maxdisp, cost_thr);
		}
	}
	else if (mxGetClassID(input[0]) == mxUINT16_CLASS && mxGetClassID(input[1]) == mxUINT16_CLASS)
	{
		MexImage<unsigned short> Left(input[0]);
		MexImage<unsigned short> Right(input[1]);
		if (Left.width != Right.width || Left.height != Right.height || Left.layers != Right.layers)
		{
			mexErrMsgTxt("Resolution of Left and Right images must coincide!!");
		}
		const size_t dims[] = { (size_t)Left.height, (size_t)Left.width, (size_t)layers };
		output[0] = mxCreateNumericArray(3, dims, mxUINT16_CLASS, mxREAL);
		
		MexImage<unsigned short> CostL(output[0]);
		MexImage<unsigned short> GradientL(Left.width, Left.height);
		MexImage<unsigned short> GradientR(Left.width, Left.height);
		#pragma omp parallel sections
		{
			#pragma omp section
			{
				compute_gradient<unsigned short>(Left, GradientL, 32767.5);
			}
			#pragma omp section
			{
				compute_gradient<unsigned short>(Right, GradientR, 32767.5);
			}
		}

		compute_cost<unsigned short>(Left, Right, GradientL, GradientR, CostL, mindisp, maxdisp, cost_thr);
		if (nout > 1)
		{
			output[1] = mxCreateNumericArray(3, dims, mxUINT16_CLASS, mxREAL);
			MexImage<unsigned short> CostR(output[1]);
			compute_cost<unsigned short>(Right, Left, GradientR, GradientL, CostR, -mindisp, -maxdisp, cost_thr);
		}
	}
	else if (mxGetClassID(input[0]) == mxSINGLE_CLASS && mxGetClassID(input[1]) == mxSINGLE_CLASS)
	{
		MexImage<float> Left(input[0]);
		MexImage<float> Right(input[1]);
		if (Left.width != Right.width || Left.height != Right.height || Left.layers != Right.layers)
		{
			mexErrMsgTxt("Resolution of Left and Right images must coincide!!");
		}
		const size_t dims[] = { (size_t)Left.height, (size_t)Left.width, (size_t)layers };
		output[0] = mxCreateNumericArray(3, dims, mxSINGLE_CLASS, mxREAL);

		MexImage<float> CostL(output[0]);
		MexImage<float> GradientL(Left.width, Left.height);
		MexImage<float> GradientR(Left.width, Left.height);
		#pragma omp parallel sections
		{
			#pragma omp section
			{
				compute_gradient<float>(Left, GradientL, 0);
			}
			#pragma omp section
			{
				compute_gradient<float>(Right, GradientR, 0);
			}
		}

		compute_cost<float>(Left, Right, GradientL, GradientR, CostL, mindisp, maxdisp, cost_thr);
		if (nout > 1)
		{
			output[1] = mxCreateNumericArray(3, dims, mxSINGLE_CLASS, mxREAL);
			MexImage<float> CostR(output[1]);
			compute_cost<float>(Right, Left, GradientR, GradientL, CostR, -mindisp, -maxdisp, cost_thr);
		}
	}
	else if (mxGetClassID(input[0]) == mxDOUBLE_CLASS && mxGetClassID(input[1]) == mxDOUBLE_CLASS)
	{
		MexImage<double> Left(input[0]);
		MexImage<double> Right(input[1]);
		if (Left.width != Right.width || Left.height != Right.height || Left.layers != Right.layers)
		{
			mexErrMsgTxt("Resolution of Left and Right images must coincide!!");
		}
		const size_t dims[] = { (size_t)Left.height, (size_t)Left.width, (size_t)layers };
		output[0] = mxCreateNumericArray(3, dims, mxDOUBLE_CLASS, mxREAL);

		MexImage<double> CostL(output[0]);
		MexImage<double> GradientL(Left.width, Left.height);
		MexImage<double> GradientR(Left.width, Left.height);
		#pragma omp parallel sections
		{
			#pragma omp section
			{
				compute_gradient<double>(Left, GradientL, 0);
			}
			#pragma omp section
			{
				compute_gradient<double>(Right, GradientR, 0);
			}
		}

		compute_cost<double>(Left, Right, GradientL, GradientR, CostL, mindisp, maxdisp, cost_thr);
		if (nout > 1)
		{
			output[1] = mxCreateNumericArray(3, dims, mxDOUBLE_CLASS, mxREAL);
			MexImage<double> CostR(output[1]);
			compute_cost<double>(Right, Left, GradientR, GradientL, CostR, -mindisp, -maxdisp, cost_thr);
		}
	}
	

}
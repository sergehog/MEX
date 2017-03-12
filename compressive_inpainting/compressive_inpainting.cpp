//compressive_inpainting
/** My implementation of the Criminisi inpainting approach
*	Extended with optional Priority map, which can define direction of inpainting
*	@file criminisi_inpainting.cpp
*	@date 11.03.2013
*	@author Sergey Smirnov <sergey.smirnov@tut.fi>
*/

#include "../common/meximage.h"
#include <float.h>
#include <cmath>
#include <atomic>
#include <algorithm>
#include <vector>
#include <string.h>

#ifdef __cplusplus 
extern "C" {
#include "mex.h"
}
#endif // __cplusplus
#ifndef _DEBUG
#include <omp.h>
#endif

#ifdef WIN32
#define isnan _isnan
#endif

typedef signed char int8;
typedef unsigned char uint8;

#pragma comment (lib, "libeng.lib") 
#pragma comment (lib, "libmx.lib")
#pragma comment (lib, "libmex.lib")

using namespace mymex;
static const float mynan = sqrt(-1.f);

template<unsigned colors>
void compressive_inpainting_radius(const int radius, const MexImage<float> &Signal, const MexImage<float> &Filled, const int search);

template<unsigned radius>
void separable_errode(MexImage<bool> &Valid);

#define diameter(radius) ((radius)*2+1)
#define window(radius) (((radius)*2+1)*((radius)*2+1))
#define ksparse 8

template<unsigned colors, unsigned radius>
void calculate_transform(const float *transform);

template<unsigned colors, unsigned radius>
void add_coefficient(MexImage<float> &Compressed, MexImage<uint8_t> &Coeffs, const float value, const int x, const int y, const int k);

template<unsigned colors, unsigned radius>
void patch2vector(MexImage<float> &Signal, const float * vector, const int x, const int y)
{
	for (int c = 0, j = 0; c < colors; c++)
		for (int dx = -radius; dx <= radius; dx++)
			for (int dy = -radius; dy <= radius; dy++, j++)
			{
				vector[j] = Signal(x + dx, y + dy, c);
			}
}


template<unsigned colors, unsigned radius>
void compressive_inpainting(const MexImage<float> &Signal, const MexImage<float> &Filled, const int search)
{
	const int width = Signal.width;
	const int height = Signal.height;
	const long HW = height * width;

	MexImage<bool> &Valid(width, height);
	for (long i = 0; i < HW; i++)
	{
		const int x = i / height;
		const int y = i % height;

		Valid[i] = !isnan(Signal[i]);
		for (int c = 0; c < colors; c++)
		{
			//Filled[i + c*HW] = Signal[i + c*HW];
			Filled(x, y, c) = Signal(x, y, c);
		}
	}
	separable_errode<radius>(Valid);

	//float vector[window(radius)*colors];
	float transform[window(radius)*colors*window(radius)*colors];
	calculate_transform<colors, radius>(transform);
	
	// "compressed patch-hash" storage for valid signal patches.
	// only ksparse largest coefficients are stored
	// otherwise one would need to store window(radius)*colors coefficients for each valid pixel 
	MexImage<float> Compressed(width, height, ksparse);
	MexImage<uint8_t> Coeffs(width, height, ksparse);
	Coeffs.setval(0);
	Compressed.setval(0.f);

	// hashing of valid patches (linear-complexity operation)
	#pragma omp parallel for
	for (long i = 0; i < HW; i++)
	{
		float vector[window(radius)*colors];
		const int x = i / height;
		const int y = i % height;
		if (Valid[i])
		{			
			patch2vector<colors, radius>(Signal, vector, x, y);
						
			// number of coefficient
			for (int k = 0; k < window(radius)*colors; k++)
			{
				const int kk = k * window(radius)*colors;
				float value = 0;
				// index of signal in the vector
				for (int j = 0; j < window(radius)*colors; k++)
				{
					value += vector[j] * transform[kk + j];
				}

				//const float abs_value = abs(value);

				add_coefficient(Compressed, Coeffs, value, x, y, k);
			}			
		}
	}

	MexImage<float> DataTerm(width, height);
	//MexImage<float> Average(width, height, colors);
	MexImage<float> GradientX(width, height);
	MexImage<float> GradientY(width, height);
	MexImage<float> IntegralConf(width, height);
	MexImage<int8> NormalX(width, height);
	MexImage<int8> NormalY(width, height);
	MexImage<bool> Source(width, height);

#pragma omp parallel sections
	{
#pragma omp section
		{
			// normals map (used to find boundary patches)	
			NormalX.setval(0);
			NormalY.setval(0);
			calculateNormal(Valid, NormalX, NormalY, 0, 0, width - 1, height - 1);
		}
#pragma omp section
		{
			//Average.setval(0.f);
			DataTerm.setval(0.f);
			Source.setval(false);
			GradientX.setval(0.f);
			GradientY.setval(0.f);
		}

#pragma omp section
		{
			// calculate fast confidence
			IntegralConf.set(Confidence);
			IntegralConf.IntegralImage(true);
		}
	}

	// fast binary map of valid patches	
	calculateSource(Valid, IntegralConf, Source, 0, 0, width, height, radius);

	//// average map (used in early termination)
	//if (earlyTerm)
	//{
	//	calculateAverage(Signal, Valid, Average, 0, 0, width - 1, height - 1, radius);
	//}

	// gradients map (isophotes for data term)
	if (useDataTerm)
	{
		if (averagedGradient)
		{
			calculateGradient(Average, Valid, GradientX, GradientY, 0, 0, width - 1, height - 1);
		}
		else
		{
			calculateGradient(Signal, Valid, GradientX, GradientY, 0, 0, width - 1, height - 1);
		}
	}

	

}

void mexFunction(int nout, mxArray* output[], int in, const mxArray* input[])
{
#ifndef _DEBUG
	omp_set_num_threads(std::max<int>(4, omp_get_max_threads()));
	omp_set_dynamic(std::max<int>(2, omp_get_max_threads() / 2));;
#endif	

	if (in < 1 || in > 8 || nout != 1 || mxGetClassID(input[0]) != mxSINGLE_CLASS)
	{
		mexPrintf("USAGE: [Filled] = compressive_inpainting(Signal, <radius, search, option1, option2, ...>);\n");
		mexPrintf("Filled - all NaN values in 'Single' replaced by predicted values.\n");
		mexPrintf("Signal - multi-color single-valued image with NaN values.\n");
		mexPrintf("search - radius of the search area. <default: 0 - whole image>\n");
		mexPrintf("radius - radius of the patch size. <default: 3>\n");
		//mexPrintf("Possible options: \n");
		//mexPrintf("'UpdateSource' - uses inpainted area as a source.\n");
		//mexPrintf("'EarlyTermination' - Tries to speed-up exhausive patch search.\n");
		//mexPrintf("'AveragedGradient' - requires 'EarlyTermination', not compatible with 'NoDataTerm'.\n");
		//mexPrintf("'NoDataTerm' - Onion-peel updating approach.\n");
		//mexPrintf("'SmallerUpdate' - Updated area is smaller than a patch size.\n");
		//mexPrintf("'I100' - Performs 100 iterations at max (number can vary).\n");
		mexErrMsgTxt("Wrong input/output parameters!");
	}

	MexImage<float> Signal(input[0]);

	const int height = Signal.height;
	const int width = Signal.width;
	const int HW = Signal.layer_size;
	const int colors = Signal.layers;

	matlab_size dims3d[] = { (matlab_size)height, (matlab_size)width, colors };
	output[0] = mxCreateNumericArray(3, dims3d, mxSINGLE_CLASS, mxREAL);
	MexImage<float> Filled(output[0]);


	const int radius = (in > 1) ? std::max(1, (int)mxGetScalar(input[1])) : 3;
	const int _search = (in > 2) ? std::abs((int)mxGetScalar(input[2])) : 0;
	const int search = _search ? _search : std::max(width, height);

	if (colors == 1)
	{
		compressive_inpainting_radius<1>(radius, Signal, Filled, search);
	}
	else if (colors == 2)
	{
		compressive_inpainting_radius<2>(radius, Signal, Filled, search);
	}
	else if (colors == 3)
	{
		compressive_inpainting_radius<3>(radius, Signal, Filled, search);
	}
	else if (colors == 4)
	{
		compressive_inpainting_radius<4>(radius, Signal, Filled, search);
	}
	else if (colors == 5)
	{
		compressive_inpainting_radius<5>(radius, Signal, Filled, search);
	}
	else 
	{
		mexErrMsgTxt("Too many colors!");
	}	

}


template<unsigned colors>
void compressive_inpainting_radius(const int radius, const MexImage<float> &Signal, const MexImage<float> &Filled, const int search)
{
	if (radius == 1)
	{
		compressive_inpainting<colors, 1>(Signal, Filled, const int search);
	}
	else if (radius == 2)
	{
		compressive_inpainting<colors, 2>(Signal, Filled, const int search);
	}
	else if (radius == 3)
	{
		compressive_inpainting<colors, 3>(Signal, Filled, const int search);
	}
	else if (radius == 4)
	{
		compressive_inpainting<colors, 4>(Signal, Filled, const int search);
	}
	else if (radius == 5)
	{
		compressive_inpainting<colors, 5>(Signal, Filled, const int search);
	}
	else if (radius == 6)
	{
		compressive_inpainting<colors, 6>(Signal, Filled, const int search);
	}
	else
	{
		mexErrMsgTxt("Too big patch radius!");
	}

}

template<unsigned radius>
void separable_errode(MexImage<bool> &Valid)
{
	const int width = Valid.width;
	const int height = Valid.height;

	for (int x = 0; x < width; x++)
	{
		for (int y = 0; y < height; y++)
		{
			int k = 0; // here we also exclude boundary pixels
			if (Valid(x, y) && k <= radius)
			{
				Valid(x, y) = 0;
				k++;
			}
			else if (!Valid(x, y))
			{
				k = 0;
			}
		}
		for (int y = height - 1; y >= 0; y--)
		{
			int k = 0;
			if (Valid(x, y) && k <= radius)
			{
				Valid(x, y) = 0;
				k++;
			}
			else if (!Valid(x, y))
			{
				k = 0;
			}
		}
	}

	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			int k = 0;
			if (Valid(x, y) && k <= radius)
			{
				Valid(x, y) = 0;
				k++;
			}
			else if (!Valid(x, y))
			{
				k = 0;
			}
		}
		for (int x = width - 1; x >= 0; x--)
		{
			int k = 0;
			if (Valid(x, y) && k <= radius)
			{
				Valid(x, y) = 0;
				k++;
			}
			else if (!Valid(x, y))
			{
				k = 0;
			}
		}
	}
}

template<unsigned colors, unsigned radius>
void calculate_transform(const float *transform)
{
	const float norm = sqrt(2.f / diameter(radius));
	// calculate transform
	for (int i = 0; i < window(radius)*colors; i++)
	{
		const int k = i * window(radius)*colors;

		for (int j = 0; j < window(radius)*colors; j++)
		{
			const int k3 = j / window(radius); //colors
			const int k2 = (j - k3*window(radius)) / diameter(radius); // spatial 1
			const int k1 = j % diameter(radius); // spatial 2
			HT norm1 = (k1 == 0) ? 1 / sqrt((HT)2) : 1;
			HT norm2 = (k2 == 0) ? 1 / sqrt((HT)2) : 1;
			HT norm3 = (k3 == 0) ? 1 / sqrt((HT)2) : 1;

			transform[k + j] = norm * norm1 * norm2 * norm3 * cos((M_PI / diameter(radius))*(n1 + 0.5)*k1) * cos((M_PI / diameter(radius))*(n2 + 0.5)*k2) * cos((M_PI / colors)*(n3 + 0.5)*k3);
		}

	}
}




template<unsigned colors, unsigned radius>
void add_coefficient(MexImage<float> &Compressed, MexImage<uint8_t> &Coeffs, const float value, const int x, const int y, const int k)
{
	float insert_value = value;
	int insertK = k;
	float abs_value = abs(value);
	for (int s = 0; s < ksparse && abs_value > 0; s++)
	{
		if (abs_value > abs(Compressed(x, y, s)))
		{
			const int tempK = Coeffs(x, y, s);
			const float tempVal = Compressed(x, y, s);
			Compressed(x, y, s) = insert_value;
			Coeffs(x, y, s) = insertK;
			insert_value = tempVal;
			abs_value = abs(tempVal);
			insertK = tempK;
		}
	}
}
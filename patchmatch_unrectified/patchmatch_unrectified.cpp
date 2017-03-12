/** Slanted PatchMatch for unrectified, but calibrated images
* @file patchmatch_unrectified
* @date 26.05.2016
* @author Sergey Smirnov
*/
#ifdef __cplusplus 
extern "C" {
#include "mex.h"
}
#endif // __cplusplus
//#include "../common/common.h"
#include "../common/meximage.h"
#include <cstdint>
#include <memory>
#include <algorithm>
#define GLM_FORCE_CXX11  
#include <glm/glm.hpp>
#ifndef _NDEBUG
#include <omp.h>
#endif


#pragma comment (lib, "libeng.lib") 
#pragma comment (lib, "libmx.lib")
#pragma comment (lib, "libmex.lib")
#pragma warning(disable:4244)
#pragma warning(disable:4018)

using namespace mymex;


template<int radius>
void prepare_weights(const MexImage<const float> &Guide, const MexImage<float> &Weights, float const sigma)
{
	const int width = Guide.width;
	const int height = Guide.height;
	const int colors = Guide.layers;
	const int64_t HW = Guide.layer_size;
	const int diameter = radius * 2 + 1;
	const int window = (radius * 2 + 1) * (radius * 2 + 1);

#pragma omp parallel for
	for (int64_t i = 0; i < HW; i++)
	{
		const int x = i / height;
		const int y = i % height;
		for (int w = 0; w < window; w++)
		{
			const int dx = w / diameter - radius;
			const int dy = w % diameter - radius;

			const int xx = x + dx;
			const int yy = y + dy;

			if (xx < 0 || xx >= width || yy < 0 || yy >= height)
			{
				continue;
			}

			float diff = 0;
			for (int c = 0; c < colors; c++)
			{
				diff += (Guide(x, y, c) - Guide(xx, yy, c)) * (Guide(x, y, c) - Guide(xx, yy, c));
			}

			Weights(x, y, w) = exp(-sqrt(diff) / sigma);
		}
	}
}

template<int radius>
void optimized_patchmatch(const MexImage<const float> & Left, const MexImage<const float> & Right, const MexImage<const float> &LeftGuide, const MexImage<const float> &RightGuide, const MexImage<float> &PlanesL, const MexImage<float> &PlanesR, const glm::mat4x3 CL, const glm::mat4x3 CR, const MexImage<float> & BestErrorL, const MexImage<float> & BestErrorR, const float minZ, const float maxZ, const float sigma, const int iterations)
{
	const int width = Left.width;
	const int height = Left.height;
	const int diameter = radius * 2 + 1;
	const int window = (radius * 2 + 1) * (radius * 2 + 1);
	const MexImage<float> WeightsL(width, height, window);
	const MexImage<float> WeightsR(width, height, window);
	#pragma omp parallel sections
	{
		#pragma omp section
		{
			WeightsL.setval(0.f);
			prepare_weights<radius>(LeftGuide, WeightsL, sigma);
		}
		#pragma omp section
		{
			WeightsR.setval(0.f);
			prepare_weights<radius>(RightGuide, WeightsR, sigma);
		}
	}
		
	#pragma omp parallel sections
	{
		#pragma omp section
		{
			single_patchmatch<radius>(CL, CR, Left, Right, WeightsL, PlanesL, BestErrorL, minZ, maxZ, iterations);
		}
		#pragma omp section
		{
			single_patchmatch<radius>(CR, CL, Right, Left, WeightsR, PlanesR, BestErrorR, minZ, maxZ, iterations);
		}
	}
}

template<int radius>
void single_patchmatch(const glm::mat4x3 CRef, const glm::mat4x3 CTmpl, const MexImage<const float> &Reference, const MexImage<const float> &Template, const MexImage<float> &Weights, const MexImage<float> &Planes, const MexImage<float> & BestError, const float minZ, const float maxZ, const float iterations)
{
	const glm::mat4 MVP = glm::mat4(CTmpl) * glm::inverse(glm::mat4(CRef));

	const int width = Reference.width;
	const int height = Reference.height;
	const int colors = Reference.layers;
	const int64_t HW = static_cast<int64_t>(Reference.layer_size);

	const int diameter = radius * 2 + 1;
	const int window = diameter*diameter;
	
	float normal_std = 10;
	const float z_mean = (maxZ + minZ) / 2.f;
	float z_std = maxZ - z_mean;

	// Random initialization
	#pragma omp parallel for	
	for (int64_t i = 0; i < HW; i++)
	{
		const int x = i / height;
		const int y = i % height;
		const float z = z_mean + z_std  * (1 - 2 * float(rand()) / (RAND_MAX));		
		const float nx = (1 - 2 * float(rand()) / (RAND_MAX));
		const float ny = (1 - 2 * float(rand()) / (RAND_MAX));
		const float nz = float(rand()) / (RAND_MAX)+0.0001;
		const float norm = sqrt(nx*nx + ny*ny + nz*nz);
		const float a = -nx / nz;
		const float b = -ny / nz;
		const float c = (nx*x + ny*y + nz*z) / nz;
		Planes(x, y, 0) = a;
		Planes(x, y, 1) = b;
		Planes(x, y, 2) = c;
		BestError(x, y) = check_error<radius>(MVP, Reference, Template, Weights, x, y, a, b, c);
	}

	for (int iter = 0; iter < iterations; iter++)
	{
		// left-to-right pass
		#pragma omp parallel for
		for (int y = 0; y < height; y++)
		{
			for (int x = 1; x < width; x++)
			{
				const float a = Planes(x - 1, y, 0);
				const float b = Planes(x - 1, y, 1);
				const float c = Planes(x - 1, y, 2);
				const float z = a*x + b*y + c;

				if (z < minZ || z > maxZ)
				{
					continue;
				}

				const float cost_value = check_error<radius>(MVP, Reference, Template, Weights, x, y, a, b, c);

				if (cost_value < BestError(x, y))
				{
					BestError(x, y) = cost_value;
					Planes(x, y, 0) = a;
					Planes(x, y, 1) = b;
					Planes(x, y, 2) = c;
				}
			}
		}

		// right-to-left pass
		#pragma omp parallel for
		for (int y = 0; y < height; y++)
		{
			for (int x = width - 2; x >= 0; x--)
			{
				const float a = Planes(x + 1, y, 0);
				const float b = Planes(x + 1, y, 1);
				const float c = Planes(x + 1, y, 2);
				const float z = a*x + b*y + c;

				if (z < minZ || z > maxZ)
				{
					continue;
				}

				const float cost_value = check_error<radius>(MVP, Reference, Template, Weights, x, y, a, b, c);

				if (cost_value < BestError(x, y))
				{
					BestError(x, y) = cost_value;
					Planes(x, y, 0) = a;
					Planes(x, y, 1) = b;
					Planes(x, y, 2) = c;
				}
			}
		}

		// up-to-bottom pass
#pragma omp parallel for
		for (int x = 0; x < width; x++)
		{
			for (int y = 1; y < height; y++)
			{
				const float a = Planes(x, y - 1, 0);
				const float b = Planes(x, y - 1, 1);
				const float c = Planes(x, y - 1, 2);
				const float z = a*x + b*y + c;

				if (z < minZ || z > maxZ)
				{
					continue;
				}

				const float cost_value = check_error<radius>(MVP, Reference, Template, Weights, x, y, a, b, c);

				if (cost_value < BestError(x, y))
				{
					BestError(x, y) = cost_value;
					Planes(x, y, 0) = a;
					Planes(x, y, 1) = b;
					Planes(x, y, 2) = c;
				}
			}
		}

		// bottom-to-up pass
#pragma omp parallel for
		for (int x = 0; x < width; x++)
		{
			for (int y = height - 2; y >= 0; y--)
			{
				const float a = Planes(x, y + 1, 0);
				const float b = Planes(x, y + 1, 1);
				const float c = Planes(x, y + 1, 2);
				const float z = a*x + b*y + c;

				if (z < minZ || z > maxZ)
				{
					continue;
				}

				const float cost_value = check_error<radius>(MVP, Reference, Template, Weights, x, y, a, b, c);

				if (cost_value < BestError(x, y))
				{
					BestError(x, y) = cost_value;
					Planes(x, y, 0) = a;
					Planes(x, y, 1) = b;
					Planes(x, y, 2) = c;
				}
			}
		}

		// refinement
#pragma omp parallel for
		for (int64_t i = 0; i < HW; i++)
		{
			const int x = i / height;
			const int y = i % height;
			const float a = Planes(x, y, 0);
			const float b = Planes(x, y, 1);
			const float c = Planes(x, y, 2);
			const float z = (a*x + b*y + c);
			float nz = sqrt(1.f / (1 + a*a + b*b));
			float nx = -a * nz;
			float ny = -b * nz;
			nx += normal_std * (1 - 2 * float(rand()) / (RAND_MAX));
			ny += normal_std * (1 - 2 * float(rand()) / (RAND_MAX));
			nz += normal_std * (1 - 2 * float(rand()) / (RAND_MAX));
			nz = (nz < 0.001) ? 0.001 : nz;
			
			const float z2 = z + z_std * (1 - 2 * float(rand()) / (RAND_MAX));
			const float norm = sqrt(nx*nx + ny*ny + nz*nz);
			const float a2 = -nx / nz;
			const float b2 = -ny / nz;
			const float c2 = (nx*x + ny*y + nz*z2) / nz;

			const float cost_value = check_error<radius>(MVP, Reference, Template, Weights, x, y, a2, b2, c2);

			if (cost_value < BestError(x, y))
			{
				BestError(x, y) = cost_value;
				Planes(x, y, 0) = a2;
				Planes(x, y, 1) = b2;
				Planes(x, y, 2) = c2;
			}
		}
		normal_std /= 2.f;
		z_std /= 2.f;
	}

}

template<int radius>
float check_error(const glm::mat4 MVP, const MexImage<const float> &Reference, const MexImage<const float> &Template, const MexImage<float> &Weights, const int x, const int y, const float a, const float b, const float c)
{
	const int width = Reference.width;
	const int height = Reference.height;
	const int colors = Reference.layers;

	const int diameter = radius * 2 + 1;
	const int window = diameter*diameter;
	

	float cost_value = 0;
	float weights = 0;
	float pixels = 0;
	for (int w = 0; w < window; w++)
	{
		const int dx = w / diameter - radius;
		const int dy = w % diameter - radius;

		const int xx = x + dx;
		const int yy = y + dy;

		if (xx < 0 || xx >= width || yy < 0 || y >= height)
		{
			continue;
		}

		pixels ++;

		const float z = a * xx + b * yy + c;

		glm::vec4 uv = MVP * glm::vec4(xx*z, yy*z, z, 1);
		const float _xr = uv.x / uv.z;
		const float _yr = uv.y / uv.z;
		const float xr = _xr < 0.f ? 0.f : (_xr >= width - 1.f ? width - 1.f : _xr);
		const float yr = _yr < 0.f ? 0.f : (_yr >= height - 1.f ? height - 1.f : _yr);
		
		const float A = xr - floor(xr);
		const float B = yr - floor(yr);
		const int xf = int(floor(xr));
		const int xc = int(ceil(xr));
		const int yf = int(floor(yr));
		const int yc = int(ceil(yr));
		
		float diff = 0.f;
		for (int c = 0; c < colors; c++)
		{
			float R_color = Template(xf, yf, c) * (1-A) * (1-B);
			R_color += Template(xc, yf, c) * A  * (1-B);
			R_color += Template(xc, yc, c) * A  * B;
			R_color += Template(xf, yc, c) * A  * (1-B);
			diff += abs(Reference(xx, yy, c) - R_color);
		}		
		
		// cost thresholding
		diff = diff > 100.f ? 100.f : diff;

		cost_value += diff * Weights(x, y, w);
		weights += Weights(x, y, w);
	}
	
	return pixels > (diameter) ? cost_value / weights : FLT_MAX;
}


void mexFunction(int nout, mxArray* output[], int in, const mxArray* input[])
{
#ifndef _NDEBUG
	omp_set_num_threads(std::max(8, omp_get_max_threads()));
	omp_set_dynamic(std::max(7, omp_get_max_threads() - 1));
	//omp_set_dynamic(omp_get_max_threads());
#endif	

	if (in < 9 || in > 11 || nout != 4 || mxGetClassID(input[0]) != mxSINGLE_CLASS || mxGetClassID(input[1]) != mxSINGLE_CLASS || mxGetClassID(input[2]) != mxSINGLE_CLASS || mxGetClassID(input[3]) != mxSINGLE_CLASS || mxGetClassID(input[4]) != mxSINGLE_CLASS || mxGetClassID(input[5]) != mxSINGLE_CLASS)
	{
		mexErrMsgTxt("USAGE: [Al, Ar, ErrL, ErrR] = patchmatch_unrectified(single(Left), single(Right), single(LeftGuide), single(RightGuide), single(CL), single(CR), radius, minZ, maxZ [, iterations=4, sigma=10]);");
	}

	const int radius = static_cast<int>(mxGetScalar(input[6]));	
	const float minZ = static_cast<float>(mxGetScalar(input[7]));
	const float maxZ = static_cast<float>(mxGetScalar(input[8]));
	const int iterations = (in > 9) ? static_cast<int>(mxGetScalar(input[9])) : 4;
	const float sigma = (in > 10) ? static_cast<int>(mxGetScalar(input[10])) : 10.f;
	
	if (maxZ <= minZ)
	{
		mexErrMsgTxt("ERROR: 'maxZ' must be larger than 'minZ'!");
	}
	if ((mxGetDimensions(input[4]))[0] != 3 || (mxGetDimensions(input[4]))[1] != 4)
	{	
		mexErrMsgTxt("Camera matrix CL must be of size 3x4. \n");
	}
	if ((mxGetDimensions(input[5]))[0] != 3 || (mxGetDimensions(input[5]))[1] != 4)
	{
		mexErrMsgTxt("Camera matrix CR must be of size 3x4. \n");
	}

	const MexImage<const float> Left(input[0]);
	const MexImage<const float> Right(input[1]);
	const MexImage<const float> LeftGuide(input[2]);
	const MexImage<const float> RightGuide(input[3]);
	const float * const cl = (float*)mxGetData(input[4]);	
	const glm::mat4x3 CL = glm::mat4x3(cl[0], cl[1], cl[2], cl[3], cl[4], cl[5], cl[6], cl[7], cl[8], cl[9], cl[10], cl[11]);
	const float * const cr = (float*)mxGetData(input[5]);
	const glm::mat4x3 CR = glm::mat4x3(cr[0], cr[1], cr[2], cr[3], cr[4], cr[5], cr[6], cr[7], cr[8], cr[9], cr[10], cr[11]);

	const int width = Left.width;
	const int height = Left.height;
	const int layers = Left.layers;
	const int colors = LeftGuide.layers;
	const int64_t HW = static_cast<int64_t>(Left.layer_size);

	if (height != Right.height || width != Right.width || layers != Right.layers)
	{
		mexErrMsgTxt("ERROR: Sizes of parameters 'Left', 'Right' must be the same.");
	}
	if (height != LeftGuide.height || width != LeftGuide.width || colors != LeftGuide.layers)
	{
		mexErrMsgTxt("ERROR: Sizes of parameters 'Left', 'Right' must be the same.");
	}
	if (height != RightGuide.height || width != RightGuide.width || colors != RightGuide.layers)
	{
		mexErrMsgTxt("ERROR: Sizes of parameters 'Left', 'Right' must be the same.");
	}

	if (colors < 1 || colors > 9)
	{
		mexErrMsgTxt("Too many colors in your images.");
	}

	const mwSize planeDims[] = { (size_t)height, (size_t)width, 3 };
	const mwSize errDims[] = { (size_t)height, (size_t)width, 1 };

	output[0] = mxCreateNumericArray(3, planeDims, mxSINGLE_CLASS, mxREAL);
	output[1] = mxCreateNumericArray(3, planeDims, mxSINGLE_CLASS, mxREAL);
	output[2] = mxCreateNumericArray(3, errDims, mxSINGLE_CLASS, mxREAL);
	output[3] = mxCreateNumericArray(3, errDims, mxSINGLE_CLASS, mxREAL);
	const MexImage<float> PlanesL(output[0]);
	const MexImage<float> PlanesR(output[1]);
	const MexImage<float> BestErrorL(output[2]);
	const MexImage<float> BestErrorR(output[3]);
	//const MexImage<float> BestErrorL(width, height);
	//const MexImage<float> BestErrorR(width, height);
	PlanesL.setval(0);
	PlanesR.setval(0);
	BestErrorL.setval(FLT_MAX);
	BestErrorR.setval(FLT_MAX);

	switch (radius)
	{
	case 2: optimized_patchmatch<2>(Left, Right, LeftGuide, RightGuide, PlanesL, PlanesR, CL, CR, BestErrorL, BestErrorR, minZ, maxZ, iterations, sigma); break;
	case 3: optimized_patchmatch<3>(Left, Right, LeftGuide, RightGuide, PlanesL, PlanesR, CL, CR, BestErrorL, BestErrorR, minZ, maxZ, iterations, sigma); break;
	case 4: optimized_patchmatch<4>(Left, Right, LeftGuide, RightGuide, PlanesL, PlanesR, CL, CR, BestErrorL, BestErrorR, minZ, maxZ, iterations, sigma); break;
	case 5: optimized_patchmatch<5>(Left, Right, LeftGuide, RightGuide, PlanesL, PlanesR, CL, CR, BestErrorL, BestErrorR, minZ, maxZ, iterations, sigma); break;
	case 6: optimized_patchmatch<6>(Left, Right, LeftGuide, RightGuide, PlanesL, PlanesR, CL, CR, BestErrorL, BestErrorR, minZ, maxZ, iterations, sigma); break;
	case 7: optimized_patchmatch<7>(Left, Right, LeftGuide, RightGuide, PlanesL, PlanesR, CL, CR, BestErrorL, BestErrorR, minZ, maxZ, iterations, sigma); break;
	case 8: optimized_patchmatch<8>(Left, Right, LeftGuide, RightGuide, PlanesL, PlanesR, CL, CR, BestErrorL, BestErrorR, minZ, maxZ, iterations, sigma); break;
	case 9: optimized_patchmatch<9>(Left, Right, LeftGuide, RightGuide, PlanesL, PlanesR, CL, CR, BestErrorL, BestErrorR, minZ, maxZ, iterations, sigma); break;
	case 10: optimized_patchmatch<10>(Left, Right, LeftGuide, RightGuide, PlanesL, PlanesR, CL, CR, BestErrorL, BestErrorR, minZ, maxZ, iterations, sigma); break;
	default: optimized_patchmatch<1>(Left, Right, LeftGuide, RightGuide, PlanesL, PlanesR, CL, CR, BestErrorL, BestErrorR, minZ, maxZ, iterations, sigma); break;
	}


}
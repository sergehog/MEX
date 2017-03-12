/** View interpolation based on view_interpolation_fast method, but can handle holes in the images
* @file view_interpolation_alpha
* @author Sergey Smirnov
* @date 22.10.2015
*/

#define GLM_FORCE_CXX11  
#include <glm/glm.hpp>
//#include <glm/gtc/matrix_transform.hpp>

#ifdef __cplusplus 
extern "C" {
#include "mex.h"
}
#endif // __cplusplus

#include "../common/meximage.h"
#include <algorithm>
#include <memory>
#include <numeric>
#ifndef _DEBUG
#include <omp.h>
#endif

#define isnan _isnan
#pragma comment (lib, "libeng.lib") 
#pragma comment (lib, "libmx.lib")
#pragma comment (lib, "libmex.lib")
#pragma warning(disable:4244)
#pragma warning(disable:4018)

using namespace mymex;
//#define colors 3
#define cost_thr 15.f
//#define DEBUG
#define MAXDIFF 255*3+1


#define LIM(A,B) ((A) < 0 ? 0 : ((A) > (B)-1 ? (B)-1 : (A)))

void projectImage(MexImage<float> &Desired, const glm::mat4 Cdinv, MexImage<float> &Image, const glm::mat4x3 Ci, const float z)
{
	const int d_width = Desired.width;
	const int d_height = Desired.height;
	const int i_width = Image.width;
	const int i_height = Image.height;
	const int colors = Image.layers;

	//const long HW = width * height;

	for (int ud = 0; ud<d_width; ud++)
	{
		for (int vd = 0; vd<d_height; vd++)
		{
			long indexd = Image.Index(ud, vd);
			const glm::vec3 uvzi = Ci * (Cdinv * glm::vec4(ud*z, vd*z, z, 1));

			const float ui = uvzi.x / uvzi.z;
			const float vi = uvzi.y / uvzi.z;
			const int ul = std::floor(ui);
			const int vl = std::floor(vi);

			const float dx = ui - ul;
			const float dy = vi - vl;

			if (ul<0 || ul >= i_width - 1 || vl < 0 || vl >= i_height - 1)
			{
				continue;
			}

			//#pragma loop(hint_parallel(colors))
			for (int c = 0; c<colors; c++)
			{
				Desired(ud, vd, c) = Image(LIM(ul, i_width), LIM(vl, i_height), c)*(1 - dx)*(1 - dy)
					+ Image(LIM(ul + 1, i_width), LIM(vl, i_height), c)*dx*(1 - dy)
					+ Image(LIM(ul, i_width), LIM(vl + 1, i_height), c)*(1 - dx)*dy
					+ Image(LIM(ul + 1, i_width), LIM(vl + 1, i_height), c)*dx*dy;
			}
		}
	}
}

void fastGaussian(const MexImage<float> &Cost, const MexImage<float> &Temporal, const float sigma)
{
	const int width = Cost.width;
	const int height = Cost.height;

	// has to be initialized with zeros
	Temporal.setval(0.f);

	// Horisontal (left-to-right & right-to-left) pass
	for (int y = 0; y<height; y++)
	{
		// left-to-right accumulation buffer
		std::unique_ptr<float[]> temporal1 = std::unique_ptr<float[]>(new float[width]);

		// right-to-left accumulation buffer
		std::unique_ptr<float[]> temporal2 = std::unique_ptr<float[]>(new float[width]);

		temporal1[0] = Cost(0, y);
		temporal2[width - 1] = Cost(width - 1, y);
		for (int x1 = 1; x1<width; x1++)
		{
			const int x2 = width - x1 - 1;
			temporal1[x1] = (Cost(x1, y) + temporal1[x1 - 1] * sigma);
			temporal2[x2] = (Cost(x2, y) + temporal2[x2 + 1] * sigma);
		}
		for (int x = 0; x<width; x++)
		{
			Temporal(x, y) = (temporal1[x] + temporal2[x]) - Cost(x, y);
		}
	}

	// Vertical (up-to-down & down-to-up) pass
	for (int x = 0; x<width; x++)
	{
		std::unique_ptr<float[]> temporal1 = std::unique_ptr<float[]>(new float[height]);
		std::unique_ptr<float[]> temporal2 = std::unique_ptr<float[]>(new float[height]);
		temporal1[0] = Temporal(x, 0);
		temporal2[height - 1] = Temporal(x, height - 1);
		for (int y1 = 1; y1<height; y1++)
		{
			const int y2 = height - y1 - 1;
			temporal1[y1] = (Temporal(x, y1) + temporal1[y1 - 1] * sigma);
			temporal2[y2] = (Temporal(x, y2) + temporal2[y2 + 1] * sigma);
		}
		for (int y = 0; y<height; y++)
		{
			Cost(x, y) = (temporal1[y] + temporal2[y]) - Temporal(x, y);
		}
	}
}


void mexFunction(int nout, mxArray* output[], int in, const mxArray* input[])
{
#ifndef _DEBUG
	omp_set_num_threads(std::max(4, omp_get_max_threads()));
	omp_set_dynamic(std::max(2, omp_get_max_threads() / 2));
#endif

	const int _offset = 5; // 5 obligatory params goes before variable numner of camera pairs

	if (in < _offset + 4 || nout != 4 || mxGetClassID(input[0]) != mxSINGLE_CLASS)
	{
		mexPrintf("View interpolation with Gaussian Aggregation\n");
		mexErrMsgTxt("USAGE: [Desired, Depth, Alpha, BestCost] = view_interpolation_fast(C_desired, minZ, maxZ, layers, sigma, single(Image1), single(Alpha1), C1, single(Image2), single(Alpha2), C2, <,..., ImageN, CN>);");
	}

	const float minZ = static_cast<float>(mxGetScalar(input[1]));
	const float maxZ = static_cast<float>(mxGetScalar(input[2]));
	const int layers = static_cast<int>(mxGetScalar(input[3]));
	const float alpha = static_cast<float>(mxGetScalar(input[4]));

	const int cameras = (in - _offset) / 3;
	const int height = mxGetDimensions(input[_offset])[0];
	const int width = mxGetDimensions(input[_offset])[1];
	const int colors = mxGetDimensions(input[_offset])[2];
	const long HW = width*height;

	if (cameras < 2)
	{
		mexErrMsgTxt("At least two camera frames must be provided!");
	}

	if ((mxGetDimensions(input[0]))[0] != 3 || (mxGetDimensions(input[0]))[1] != 4)
	{
		mexErrMsgTxt("C_desired matrix must be 3x4");
	}

	const float * const cd = (float*)mxGetData(input[0]);

	// Camera matrix of the desired view
	const glm::mat4x3 Cd = glm::mat4x3(cd[0], cd[1], cd[2], cd[3], cd[4], cd[5], cd[6], cd[7], cd[8], cd[9], cd[10], cd[11]);

	// Inverse of desired view camera matrix
	const glm::mat4 Cdinv = glm::inverse(glm::mat4(Cd));

	// double-check input data
	for (int n = 0; n<cameras; n++)
	{
		if (mxGetClassID(input[_offset + n * 3]) != mxSINGLE_CLASS)
		{
			char buff[1000];
			sprintf(buff, "Camera frame %d must be of a SINGLE-type ", n + 1);
			mexErrMsgTxt(buff);
		}

		if ((mxGetDimensions(input[_offset + n * 3]))[0] != height || (mxGetDimensions(input[_offset + n * 3]))[1] != width || (mxGetDimensions(input[_offset + n * 3]))[2] != colors)
		{
			mexErrMsgTxt("All camera frames must have the same resolution and three colors!");
		}

		if ((mxGetDimensions(input[_offset + n * 3 + 1]))[0] != height || (mxGetDimensions(input[_offset + n * 3 + 1]))[1] != width)
		{
			mexErrMsgTxt("All Alpha frames must have the same resolution!");
		}

		if (mxGetClassID(input[_offset + n * 3 + 2]) != mxSINGLE_CLASS)
		{
			char buff[1000];
			sprintf(buff, "Camera matrix %d must be of a SINGLE-type ", n + 1);
			mexErrMsgTxt(buff);
		}

		if ((mxGetDimensions(input[_offset + n * 3 + 2]))[0] != 3 || (mxGetDimensions(input[_offset + n * 3 + 2]))[1] != 4)
		{
			char buff[1000];
			sprintf(buff, "Camera matrix %d must be of sized 3x4", n + 1);
			mexErrMsgTxt(buff);
		}
	}

	std::unique_ptr<MexImage<float>*[]> Images(new MexImage<float>*[cameras]);
	std::unique_ptr<MexImage<float>*[]> Alphas(new MexImage<float>*[cameras]);
	std::unique_ptr<glm::mat4x3[]> C(new glm::mat4x3[cameras]);
	std::unique_ptr<float[]> distances(new float[cameras]);

#pragma omp parallel for
	for (int n = 0; n<cameras; n++)
	{
		Images[n] = new MexImage<float>(input[_offset + n * 3]);
		Alphas[n] = new MexImage<float>(input[_offset + n * 3 + 1]);

		const float * const c1 = (float*)mxGetData(input[_offset + n * 3 + 2]);
		C[n] = glm::mat4x3(c1[0], c1[1], c1[2], c1[3], c1[4], c1[5], c1[6], c1[7], c1[8], c1[9], c1[10], c1[11]);
		distances.get()[n] = sqrt((c1[9] - cd[9])*(c1[9] - cd[9]) + (c1[10] - cd[10])*(c1[10] - cd[10]) + (c1[11] - cd[11])*(c1[11] - cd[11]));
	}

	const float nan = sqrt(-1.f);
	const mwSize dimC[] = { (unsigned)height, (unsigned)width, (unsigned)colors };
	const mwSize dims[] = { (unsigned)height, (unsigned)width, 1 };

	//Matlab-allocated variables
	output[0] = mxCreateNumericArray(3, dimC, mxSINGLE_CLASS, mxREAL);
	output[1] = mxCreateNumericArray(3, dims, mxSINGLE_CLASS, mxREAL);
	output[2] = mxCreateNumericArray(3, dims, mxSINGLE_CLASS, mxREAL);
	output[3] = mxCreateNumericArray(3, dims, mxSINGLE_CLASS, mxREAL);

	MexImage<float> Desired(output[0]); // desired color view
	MexImage<float> Depth(output[1]); // estimated depth in the desired view
	MexImage<float> AlphaOut(output[2]); // estimated alpha
	MexImage<float> BestCost(output[3]); // estimated best cost

	BestCost.setval(FLT_MAX);
	const float all_distances = std::accumulate(&distances[0], &distances[cameras], 0.f);
	//mexPrintf("all_ditances=%f; distance[0]=%f; distance[1]=%f; cameras=%d\n", all_distances, distances[0], distances[1], cameras);
	// for each depth-layer in the space
#pragma omp parallel 
	{
		// Cost of the current depth layer
		MexImage<float> Cost(width, height);

		// Normalization field, required to predict proper cost in the invisible (NaN) areas
		MexImage<float> Normalization(width, height);

		// Temporal buffer, required for gaussian filtering
		MexImage<float> Temporal(width, height);

		// Projected images are stored here
		std::unique_ptr<MexImage<float> *[]> Colors = std::unique_ptr<MexImage<float> *[]>(new MexImage<float>*[cameras]);
		// Projected Alphas
		std::unique_ptr<MexImage<float> *[]> PAlphas = std::unique_ptr<MexImage<float> *[]>(new MexImage<float>*[cameras]);

		for (int n = 0; n<cameras; n++)
		{
			Colors[n] = new MexImage<float>(width, height, colors);
			PAlphas[n] = new MexImage<float>(width, height);
		}

		// color value will be mixed from all available cameras, if visible
		std::unique_ptr<float[]> updated_color = std::unique_ptr<float[]>(new float[colors]);

#pragma omp for		
		for (int d = 0; d < layers; d++)
		{
			const float z = 1.f / ((float(d) / layers)*(1.f / minZ - 1.f / maxZ) + 1.f / maxZ);

			Cost.setval(0.f);
			Normalization.setval(0.f);

			// project all images to a position of a desired camera O(N*L) complexity
#pragma omp parallel for	
			for (int n = 0; n < cameras; n++)
			{
				MexImage<float> &Color = *Colors[n];
				MexImage<float> &Image = *Images[n];
				MexImage<float> &Alpha = *Alphas[n];
				MexImage<float> &PAlpha = *PAlphas[n];
				Color.setval(nan);

				projectImage(Color, Cdinv, Image, C[n], z);
				projectImage(PAlpha, Cdinv, Alpha, C[n], z);
			}

			for (int n1 = 0; n1 < cameras; n1++)
			{
				MexImage<float> &Color1 = *Colors[n1];
				MexImage<float> &PAlpha1 = *PAlphas[n1];
				for (int n2 = n1 + 1; n2<cameras; n2++)
				{
					MexImage<float> &Color2 = *Colors[n2];
					MexImage<float> &PAlpha2 = *PAlphas[n2];
					for (long i = 0; i < HW; i++)
					{
						float value = 0;
						if (!isnan(Color1[i]) && !isnan(Color2[i]))
						{
							for (int c = 0; c < colors; c++)
							{
								value += abs(Color1[i + c*HW] - Color2[i + c*HW]);
							}
							//const float alpha = (PAlpha1[i] + PAlpha2[i]) / 2;
							const float alpha = PAlpha1[i] < PAlpha2[i] ? PAlpha1[i] : PAlpha2[i];
							value /= colors;
							value = value > cost_thr ? cost_thr : value;
							value = value*alpha + cost_thr*(1 - alpha)/2;
							Cost[i] += value;
							Normalization[i] += 1.f;
						}
					}
				}
			}

			fastGaussian(Cost, Temporal, alpha);
			fastGaussian(Normalization, Temporal, alpha);

#pragma omp parallel for		
			for (long i = 0; i<HW; i++)
			{
				const int x = i / height;
				const int y = i % height;
				const float cost = Cost[i] / Normalization[i];

				if (cost < BestCost[i])
				{
					for (int c = 0; c<colors; c++)
					{
						updated_color[c] = 0;
					}

					float weights = 0;
					float alpha = nan;
					for (int n = 0; n<cameras; n++)
					{
						MexImage<float> &Color = *Colors[n];
						MexImage<float> &PAlpha = *PAlphas[n];
						const float w = 1 - distances[n] / all_distances;
						if (!isnan(Color(x, y, 0)))
						{
							weights += w;
							for (int c = 0; c<colors; c++)
							{
								updated_color[c] += Color(x, y, c)*w;
							}
							alpha = isnan(alpha) || PAlpha[i] < alpha ? PAlpha[i] : alpha;
						}
						
					}

					for (int c = 0; c<colors; c++)
					{
						updated_color[c] /= weights;
					}
					//alpha /= weights;

					while (cost < BestCost[i])
					{
#pragma omp critical
						{
							for (int c = 0; c<colors; c++)
							{
								Desired(x, y, c) = updated_color[c];
							}
							AlphaOut[i] = alpha;
							Depth[i] = z;
							BestCost[i] = cost;
						}
					}
				}
			}
		}

		//clean-up projected images array
		for (int n = 0; n<cameras; n++)
		{
			delete Colors[n];
		}
	}


	//clean-up camera images array
#pragma omp parallel for
	for (int i = 0; i<cameras; i++)
	{
		delete Images[i];
	}
}
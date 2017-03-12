/** Interpolates desired view between (number of) given calibrated camera images
* @file view_interpolation_trinoc.cpp
* @author Sergey Smirnov <sergey.smirnov@tut.fi> / 3D Media Group / Tampere University of Technology
* @date 27.04.2015
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
#define colors 3
#define clralpha (colors+1)
#define cameras 2
#define cost_thr (3*15.f)
//#define DEBUG
#define MAXDIFF 255*3+1



#define LIM(A,B) ((A) < 0 ? 0 : ((A) > (B)-1 ? (B)-1 : (A)))

void projectImage(MexImage<float> &Desired, const glm::mat4 Cdinv, MexImage<float> &Image, const glm::mat4x3 Ci, const float z)
{
	const int d_width = Desired.width;
	const int d_height = Desired.height;
	const int i_width = Image.width;
	const int i_height = Image.height;
	//const int colors = Image.layers;

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
			for (int c = 0; c<clralpha; c++)
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

	Temporal.setval(0.f);

	// Horisontal (left-to-right & right-to-left) pass
	for (int y = 0; y<height; y++)
	{
		std::unique_ptr<float[]> temporal1 = std::unique_ptr<float[]>(new float[width]);
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


#define epsilon 0.01

void mexFunction(int nout, mxArray* output[], int in, const mxArray* input[])
{
#ifndef _DEBUG
	omp_set_num_threads(std::max(4, omp_get_max_threads()));
	omp_set_dynamic(std::max(2, omp_get_max_threads() / 2));
#endif

	const int _offset = 5; // obligatory params goes before variable numner of cameras

	if (in != _offset + 4 || nout != 6 || mxGetClassID(input[0]) != mxSINGLE_CLASS)
	{
		mexPrintf("View interpolation with Bilateral Cost Aggregation\n");
		mexErrMsgTxt("USAGE: [DesiredBG, DesiredFG, DepthBG, DepthFG, CostBG, CostFG] = view_interpolation(C_desired, minZ, maxZ, layers, sigma_spatial, Image1, C1, Image2, C2);");
	}

	const float minZ = static_cast<float>(mxGetScalar(input[1]));
	const float maxZ = static_cast<float>(mxGetScalar(input[2]));
	const int layers = static_cast<int>(mxGetScalar(input[3]));	
	const float sigma_spatial = std::max(0.f, std::min(1.f, static_cast<float>(mxGetScalar(input[4]))));
	
	const int height = mxGetDimensions(input[_offset])[0];
	const int width = mxGetDimensions(input[_offset])[1];
	//const int colors = mxGetDimensions(input[_offset])[2];
	const long HW = width*height;

	const float * const cd = (float*)mxGetData(input[0]);
	const glm::mat4x3 Cd = glm::mat4x3(cd[0], cd[1], cd[2], cd[3], cd[4], cd[5], cd[6], cd[7], cd[8], cd[9], cd[10], cd[11]);
	const glm::mat4 Cdinv = glm::inverse(glm::mat4(Cd));

	// double-check input data
	for (int n = 0; n<cameras; n++)
	{
		if (mxGetClassID(input[_offset + n * 2]) != mxSINGLE_CLASS)
		{
			char buff[1000];
			sprintf(buff, "Camera frame %d must be of a SINGLE-type ", n + 1);
			mexErrMsgTxt(buff);
		}

		if ((mxGetDimensions(input[_offset + n * 2]))[0] != height || (mxGetDimensions(input[_offset + n * 2]))[1] != width || (mxGetDimensions(input[_offset + n * 2]))[2] != clralpha)
		{
			mexErrMsgTxt("All images must have the same resolution and RGBA components!");
		}

		if (mxGetClassID(input[_offset + n * 2 + 1]) != mxSINGLE_CLASS)
		{
			char buff[1000];
			sprintf(buff, "Camera matrix %d must be of a SINGLE-type ", n + 1);
			mexErrMsgTxt(buff);
		}

		if ((mxGetDimensions(input[_offset + n * 2 + 1]))[0] != 3 || (mxGetDimensions(input[_offset + n * 2 + 1]))[1] != 4)
		{
			char buff[1000];
			sprintf(buff, "Camera matrix %d must be of sized 3x4", n + 1);
			mexErrMsgTxt(buff);
		}
	}
	//MexImage<float>** Images = new MexImage<float>*[cameras];
	//glm::mat4x3 *C = new glm::mat4x3[cameras];
	//glm::mat4 *Cinv = new glm::mat4[cameras];
	std::unique_ptr<MexImage<float>*[]> Images(new MexImage<float>*[cameras]);
	std::unique_ptr<glm::mat4x3[]> C(new glm::mat4x3[cameras]);
	std::unique_ptr<float[]> distances(new float[cameras]);

#pragma omp parallel for
	for (int n = 0; n<cameras; n++)
	{
		Images[n] = new MexImage<float>(input[_offset + n * 2]);
		//Colors[n] = new MexImage<float>(width, height, colors);

		const float * const c1 = (float*)mxGetData(input[_offset + n * 2 + 1]);
		C[n] = glm::mat4x3(c1[0], c1[1], c1[2], c1[3], c1[4], c1[5], c1[6], c1[7], c1[8], c1[9], c1[10], c1[11]);
		//Cinv[n] = glm::inverse(glm::mat4(C[n]));
		distances[n] = sqrt((c1[9] - cd[9])*(c1[9] - cd[9]) + (c1[10] - cd[10])*(c1[10] - cd[10]) + (c1[11] - cd[11])*(c1[11] - cd[11]));
	}

	const float nan = sqrt(-1.f);

	const mwSize dimC[] = { (unsigned)height, (unsigned)width, (unsigned)clralpha };
	const mwSize dims[] = { (unsigned)height, (unsigned)width, 1 };

	//Matlab-allocated variables
	output[0] = mxCreateNumericArray(3, dimC, mxSINGLE_CLASS, mxREAL);
	output[1] = mxCreateNumericArray(3, dimC, mxSINGLE_CLASS, mxREAL);
	output[2] = mxCreateNumericArray(3, dims, mxSINGLE_CLASS, mxREAL);
	output[3] = mxCreateNumericArray(3, dims, mxSINGLE_CLASS, mxREAL);
	output[4] = mxCreateNumericArray(3, dims, mxSINGLE_CLASS, mxREAL);
	output[5] = mxCreateNumericArray(3, dims, mxSINGLE_CLASS, mxREAL);
	
	MexImage<float> DesiredBG(output[0]); // desired color view
	MexImage<float> DesiredFG(output[1]); // desired color view
	MexImage<float> DepthBG(output[2]); // estimated depth in the desired view
	MexImage<float> DepthFG(output[3]); // estimated depth in the desired view
	MexImage<float> BestCostBG(output[4]); // estimated best cost
	MexImage<float> BestCostFG(output[5]); // estimated best cost

	//MexImage<float> DesiredBG(width, height, 3);
	//MexImage<float> DesiredFG(width, height, 4);
	//MexImage<float> BestCostBG(width, height);
	//MexImage<float> BestCostFG(width, height);
	//BestCost.setval(cost_thr + 1);
	BestCostBG.setval(cost_thr + 1);
	BestCostFG.setval(cost_thr + 1);

	DesiredBG.setval(nan);
	DesiredFG.setval(nan);
	// for each depth-layer in the space
#pragma omp parallel 
	{
		MexImage<float> CostBG(width, height);
		MexImage<float> CostFG(width, height);
		MexImage<float> NormBG(width, height);
		MexImage<float> NormFG(width, height);
		//MexImage<float> Weights(width, height);
		//MexImage<float> Weights2(width, height);
		MexImage<float> Temporal(width, height);

#pragma omp for		
		for (int d = 0; d < layers; d++)
		{
			float z = 1.f / ((float(d) / layers)*(1.f / minZ - 1.f / maxZ) + 1.f / maxZ);

			MexImage<float> Color1(width, height, clralpha);
			MexImage<float> Color2(width, height, clralpha);

			//for (int n = 0; n < cameras; n += 2)
			const int n = 0;
			{
				CostBG.setval(0.f);
				CostFG.setval(0.f);
				NormBG.setval(0.f);
				NormFG.setval(0.f);
				//Aggregated.setval(0.f);
				MexImage<float> &Image1 = *Images[n];
				MexImage<float> &Image2 = *Images[n + 1];
				Color1.setval(nan);
				projectImage(Color1, Cdinv, Image1, C[n], z);
				Color2.setval(nan);
				projectImage(Color2, Cdinv, Image2, C[n + 1], z);

				#pragma omp parallel for
				for (long i = 0; i < HW; i++)
				{
					if (isnan(Color1[i]) || isnan(Color2[i]))
					{
						//CostBG[i] = 0;
						//CostFG[i] = 0;
						continue;
					}
					const int x = i / height;
					const int y = i % height;
					
					const bool is1BG = Color1(x, y, 3) <= epsilon;
					const bool is1FG = Color1(x, y, 3) >= 1.0 - epsilon;
					const bool is2BG = Color2(x, y, 3) <= epsilon;
					const bool is2FG = Color2(x, y, 3) >= 1.0 - epsilon;
					
					float diff = 0.f;
					for (int c = 0; c < colors; c++)
					{
						diff += abs(Color1[i + c*HW] - Color2[i + c*HW]);
					}
					//diff /= colors;
					diff = diff > cost_thr ? cost_thr : diff;

					const float wBG = (is1BG && is2BG) ? 1.0f : (is1FG && is2FG) ? 0.f : 0.01f;
					const float wFG = (is1FG && is2FG) ? 1.0f : (is1BG && is2BG) ? 0.f : 0.01f;
										
					CostBG[i] = diff*wBG;
					CostFG[i] = diff*wFG;
					NormBG[i] = wBG;
					NormFG[i] = wFG;
				}

				fastGaussian(CostBG, Temporal, sigma_spatial);
				fastGaussian(NormBG, Temporal, sigma_spatial);
				fastGaussian(CostFG, Temporal, sigma_spatial);
				fastGaussian(NormFG, Temporal, sigma_spatial);

				//const float w1 = 1 - distances.get()[n] / (distances.get()[n] + distances.get()[n + 1]);
				//const float w2 = 1 - distances.get()[n + 1] / (distances.get()[n] + distances.get()[n + 1]);

#pragma omp parallel for
				for (long i = 0; i < HW; i++)
				{
					const int x = i / height;
					const int y = i % height;

					const float bgCost = CostBG[i] / NormBG[i];
					const float fgCost = CostFG[i] / NormFG[i];
					bool valid1 = !isnan(Color1(x, y, 0));
					bool valid2 = !isnan(Color2(x, y, 0));
					//bool is1BG = Color1(x, y, 4) < 0.2;
					//bool is1FG = Color1(x, y, 4) > 0.8;
					//bool is2BG = Color2(x, y, 4) < 0.2;
					//bool is2FG = Color2(x, y, 4) > 0.8;

					//const float w1BG = float(valid1 && Color1(x, y, 3) < 0.01);
					//const float w2BG = float(valid2 && Color2(x, y, 3) < 0.01);

					if (bgCost < BestCostBG(x, y))
					{						
						BestCostBG(x, y) = bgCost;
						DepthBG(x, y) = z;

						if (valid1 && valid2)
						{
							for (int c = 0; c < 3; c++)
							{
								DesiredBG(x, y, c) = Color1(x, y, 3) < Color2(x, y, 3) ? Color1(x, y, c) : Color2(x, y, c);  //(Color1(x, y, c)*w1BG + Color2(x, y, c)*w2BG) / (w1BG + w2BG);
								DesiredBG(x, y, 3) = 1;
							}
						}
						else if (valid1)
						{
							for (int c = 0; c < 3; c++)
							{
								DesiredBG(x, y, c) = Color1(x, y, c);
								DesiredBG(x, y, 3) = 1;
							}
						}
						else if (valid2)
						{
							for (int c = 0; c < 3; c++)
							{
								DesiredBG(x, y, c) = Color2(x, y, c);
								DesiredBG(x, y, 3) = 1;
							}
						}
						else 
						{								
							for (int c = 0; c < 3; c++)
							{
								DesiredBG(x, y, c) = nan;
							}
							DesiredBG(x, y, 3) = 0;
						}						
					}

					const float w1FG = float(valid1) * (Color1(x, y, 3));
					const float w2FG = float(valid2) * (Color2(x, y, 3));

					if (fgCost < BestCostFG(x, y))
					{
						BestCostFG(x, y) = fgCost;
						DepthFG(x, y) = z;

						if (valid1 || valid2)
						{
							for (int c = 0; c < 4; c++)
							{
								DesiredFG(x, y, c) = (Color1(x, y, c)*w1FG + Color2(x, y, c)*w2FG) / (w1FG + w2FG);
							}
						}
						else
						{
							for (int c = 0; c < 4; c++)
							{
								DesiredBG(x, y, c) = 0;
							}
						}
					}
					
				}
			}
		}
	}


	//clean-up camera images array
	#pragma omp parallel for
	for (int i = 0; i<cameras; i++)
	{
		delete Images[i];
	}

	//delete[] Images;

	// delete camera matrix array
	//delete[] C;

}
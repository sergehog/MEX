/** Extension of a view-interpolation mwthod to work well near discontinuities
* @file view_interpolation_layered
* @author Sergey Smirnov
* @date 1.06.2016
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
//#define cost_thr 10.f
//#define DEBUG
//#define MAXDIFF 255*3+1
//#define alpha2 0.5f

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

			// uncomment if wanna to have holes 
			//if (ul<0 || ul >= i_width - 1 || vl < 0 || vl >= i_height - 1)
			//{
			//	continue;
			//}

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
template<int colors>
struct color_values
{
	float values[colors];
};

template<int colors>
color_values<colors> projectMeColor(const glm::mat4 Cdinv, MexImage<float> &Image, const glm::mat4x3 Ci, const float z, const int u, const int v)
{
	const int i_width = Image.width;
	const int i_height = Image.height;

	color_values<colors> values;	
	const glm::vec3 uvzi = Ci * (Cdinv * glm::vec4(u*z, v*z, z, 1));

	const float ui = uvzi.x / uvzi.z;
	const float vi = uvzi.y / uvzi.z;
	const int ul = std::floor(ui);
	const int vl = std::floor(vi);

	const float dx = ui - ul;
	const float dy = vi - vl;

	// uncomment if wanna to have holes 
	//if (ul<0 || ul >= i_width - 1 || vl < 0 || vl >= i_height - 1)
	//{
	//	continue;
	//}

	//#pragma loop(hint_parallel(colors))
	for (int c = 0; c<colors; c++)
	{
		values.values[c] = Image(LIM(ul, i_width), LIM(vl, i_height), c)*(1 - dx)*(1 - dy)
			+ Image(LIM(ul + 1, i_width), LIM(vl, i_height), c)*dx*(1 - dy)
			+ Image(LIM(ul, i_width), LIM(vl + 1, i_height), c)*(1 - dx)*dy
			+ Image(LIM(ul + 1, i_width), LIM(vl + 1, i_height), c)*dx*dy;
	}
	return values;
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

	const int _offset = 7; // 7 obligatory params goes before variable numner of camera pairs

	if (in < _offset + 4 || nout != 3 || mxGetClassID(input[0]) != mxSINGLE_CLASS)
	{
		mexPrintf("View interpolation with Gaussian Aggregation\n");
		mexErrMsgTxt("USAGE: [Desired, Depth, BestCost] = view_interpolation_layered(C_desired, minZ, maxZ, layers, sigma, alpha, cost_thr, single(Image1), C1, single(Image2), C2, <,..., ImageN, CN>);");
	}

	const float minZ = static_cast<float>(mxGetScalar(input[1]));
	const float maxZ = static_cast<float>(mxGetScalar(input[2]));
	const int layers = static_cast<int>(mxGetScalar(input[3]));
	const float alpha = static_cast<float>(mxGetScalar(input[4]));
	const float alpha2 = static_cast<float>(mxGetScalar(input[5]));
	const float cost_thr = static_cast<float>(mxGetScalar(input[6]));

	const int cameras = (in - _offset) / 2;
	const int height = mxGetDimensions(input[_offset])[0];
	const int width = mxGetDimensions(input[_offset])[1];
	const int colors = mxGetDimensions(input[_offset])[2];
	const long HW = width*height;

	if (cameras != 2)
	{
		mexErrMsgTxt("two camera frames must be provided!");
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
		if (mxGetClassID(input[_offset + n * 2]) != mxSINGLE_CLASS)
		{
			char buff[1000];
			sprintf(buff, "Camera frame %d must be of a SINGLE-type ", n + 1);
			mexErrMsgTxt(buff);
		}

		if ((mxGetDimensions(input[_offset + n * 2]))[0] != height || (mxGetDimensions(input[_offset + n * 2]))[1] != width || (mxGetDimensions(input[_offset + n * 2]))[2] != colors)
		{
			mexErrMsgTxt("All camera frames must have the same resolution and three colors!");
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

	std::unique_ptr<MexImage<float>*[]> Images(new MexImage<float>*[cameras]);
	std::unique_ptr<glm::mat4x3[]> C(new glm::mat4x3[cameras]);
	std::unique_ptr<float[]> distances(new float[cameras]);

#pragma omp parallel for
	for (int n = 0; n<cameras; n++)
	{
		Images[n] = new MexImage<float>(input[_offset + n * 2]);

		const float * const c1 = (float*)mxGetData(input[_offset + n * 2 + 1]);
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

	MexImage<float> Desired(output[0]); // desired color view
	MexImage<float> Depth(output[1]); // estimated depth in the desired view
	MexImage<float> BestCost(output[2]); // estimated best cost
	MexImage<int> DepthFg(width, height);
	MexImage<int> DepthBg(width, height);	
	MexImage<float> CostFg(width, height);
	MexImage<float> CostBg(width, height);
	BestCost.setval(FLT_MAX);
	const float all_distances = std::accumulate(&distances[0], &distances[cameras], 0.f);
	

	// Create a 3D Cost Volume at first
	MexImage<float> Volume(width, height, layers);
	#pragma omp parallel 
	{
		MexImage<float> Color0(width, height, colors);
		MexImage<float> Color1(width, height, colors);

		#pragma omp for		
		for (int d = 0; d < layers; d++)
		{
			const float z = 1.f / ((float(d) / layers)*(1.f / minZ - 1.f / maxZ) + 1.f / maxZ);
			
			MexImage<float> &Image0 = *Images[0];			
			//Color0.setval(nan);
			projectImage(Color0, Cdinv, Image0, C[0], z);

			MexImage<float> &Image1 = *Images[1];
			//Color1.setval(nan);
			projectImage(Color1, Cdinv, Image1, C[1], z);

			for (long i = 0; i < HW; i++)
			{		
				const int x = i / height;
				const int y = i % height;
				float value = 0;
				for (int c = 0; c < colors; c++)
				{
					value += abs(Color0[i + c*HW] - Color1[i + c*HW]);
				}
				
				value /= colors;
				//Volume(x,y,d) = value < cost_thr ? value : cost_thr;
				Volume(x, y, d) = value;
			}		
		}
	}

	// Normalization field, required for proper recursive filtering
	MexImage<float> Normalization(width, height);
	Normalization.setval(1.f);
	fastGaussian(Normalization, Depth, alpha);

	#pragma omp parallel
	{
		// Cost of the current depth layer
		MexImage<float> Cost(width, height);
		MexImage<float> Temporal(width, height);

		#pragma omp for schedule(dynamic)	
		for (int df = layers-1; df >= 0; df--)
		{
			
			//Cost.setval(FLT_MAX);
			for (int db = df; db >= 0; db--)
			{	
				#pragma omp parallel for
				for (long i = 0; i < HW; i++)
				{
					const int x = i / height;
					const int y = i % height;
					Cost(x, y) = Volume(x, y, df) < Volume(x, y, db) ? Volume(x, y, df) : Volume(x, y, db);
					Cost(x, y) = Cost(x, y) > cost_thr ? cost_thr : Cost(x, y);
				}
				fastGaussian(Cost, Temporal, alpha);
				#pragma omp parallel for
				for (long i = 0; i < HW; i++)
				{
					const int x = i / height;
					const int y = i % height;
					const float cost = Cost(x, y) / Normalization(x, y);
					if (cost < BestCost(x, y))
					{
						BestCost(x, y) = cost;
						DepthFg(x, y) = df;
						DepthBg(x, y) = db;
						CostFg(x, y) = Volume(x, y, df);
						CostBg(x, y) = Volume(x, y, db);
					}
				}
			}

		}
	}

	fastGaussian(CostFg, Depth, alpha2);
	fastGaussian(CostBg, Depth, alpha2);
	Normalization.setval(1.f);
	fastGaussian(Normalization, Depth, alpha2);

	MexImage<float> &Image0 = *Images[0];
	MexImage<float> &Image1 = *Images[1];

	// compensate for Background holes
	MexImage<float> ColorBg(width, height, colors);
	MexImage<float> Normalization2(width, height);
	// at first, find 100% correct background image
	#pragma omp parallel for
	for (long i = 0; i < HW; i++)
	{
		const int x = i / height;
		const int y = i % height;
		const int db = DepthBg(x, y);
		const float zb = 1.f / ((float(db) / layers)*(1.f / minZ - 1.f / maxZ) + 1.f / maxZ);
		color_values<3> bg0 = projectMeColor<3>(Cdinv, Image0, C[0], zb, x, y);
		color_values<3> bg1 = projectMeColor<3>(Cdinv, Image1, C[1], zb, x, y); 
		const float costFg = CostFg(x, y) / Normalization(x, y);
		const float costBg = CostBg(x, y) / Normalization(x, y);
		
				
		if (costFg > cost_thr && costBg < cost_thr)
		{
			for (int c = 0; c < 3; c++)
			{
				ColorBg(x, y, c) = (bg0.values[c] + bg1.values[c]) / 2;				
			}
			Normalization2(x, y) = 1.f;
		}
		else
		{
			for (int c = 0; c < 3; c++)
			{
				ColorBg(x, y, c) = 0.f;
			}
			Normalization2(x, y) = 0.f;
		}
	}

	//// fill holes in the background image
	//#pragma omp parallel sections
	//{
	//	#pragma omp section
	//	{
	//		fastGaussian(MexImage<float>(ColorBg, 0), MexImage<float>(Desired, 0), alpha2);
	//	}
	//	#pragma omp section
	//	{
	//		fastGaussian(MexImage<float>(ColorBg, 1), MexImage<float>(Desired, 1), alpha2);
	//	}
	//	#pragma omp section
	//	{
	//		fastGaussian(MexImage<float>(ColorBg, 2), MexImage<float>(Desired, 2), alpha2);
	//	}
	//	#pragma omp section
	//	{
	//		fastGaussian(Normalization2, Depth, alpha2);
	//	}
	//}

	// restore valid background, but within the holes select color which is closer to the averaged one 
	#pragma omp parallel for
	for (long i = 0; i < HW; i++)
	{
		const int x = i / height;
		const int y = i % height;
		const int db = DepthBg(x, y);
		const float zb = 1.f / ((float(db) / layers)*(1.f / minZ - 1.f / maxZ) + 1.f / maxZ);
		color_values<3> bg0 = projectMeColor<3>(Cdinv, Image0, C[0], zb, x, y);
		color_values<3> bg1 = projectMeColor<3>(Cdinv, Image1, C[1], zb, x, y); 
		const float costFg = CostFg(x, y) / Normalization(x, y);
		const float costBg = CostBg(x, y) / Normalization(x, y);

		if (costFg > cost_thr && costBg < cost_thr)
		{
			// restore correct Bg
			for (int c = 0; c < 3; c++)
			{
				ColorBg(x, y, c) = (bg0.values[c] + bg1.values[c]) / 2;
			}			
		}
		else
		{
			for (int c = 0; c < 3; c++)
			{
				//ColorBg(x, y, c) /= float(Normalization2(x, y));				
				ColorBg(x, y, c) = nan;
			}

			//// chose what is better and stick with it
			//float diff0 = 0; 
			//float diff1 = 0;

			//for (int c = 0; c < 3; c++)
			//{
			//	const float average_color = ColorBg(x, y, c) / float(Normalization2(x, y));
			//	diff0 += std::abs(average_color - bg0.values[c]);
			//	diff1 += std::abs(average_color - bg1.values[c]);
			//}

			//if (diff0 < diff1)
			//{
			//	for (int c = 0; c < 3; c++)
			//	{
			//		ColorBg(x, y, c) = bg0.values[c];
			//	}
			//}
			//else
			//{
			//	for (int c = 0; c < 3; c++)
			//	{
			//		ColorBg(x, y, c) = bg0.values[c];
			//	}
			//}
		}
	}

	#pragma omp parallel for
	for (long i = 0; i < HW; i++)
	{
		const int x = i / height;
		const int y = i % height;

		const int df = DepthFg(x, y);
		const int db = DepthBg(x, y);

		const float zf = 1.f / ((float(df) / layers)*(1.f / minZ - 1.f / maxZ) + 1.f / maxZ);
		
		const float costFg = CostFg(x, y) / Normalization(x, y);
		const float costBg = CostBg(x, y) / Normalization(x, y);

		color_values<3> fg0 = projectMeColor<3>(Cdinv, Image0, C[0], zf, x, y);		
		color_values<3> fg1 = projectMeColor<3>(Cdinv, Image1, C[1], zf, x, y);
		
		
		for (int c = 0; c < 3; c++)
		{
			Desired(x, y, c) = ColorBg(x,y,c);
		}

		//if (costFg < cost_thr && costFg < costBg)
		//{
		//	//Depth(x,y) = zf;
		//	Depth(x, y) = df;
		//	BestCost(x,y) = costFg;
		//	for (int c = 0; c < 3; c++)
		//	{
		//		Desired(x, y, c) = (fg0.values[c] + fg1.values[c])/2;
		//		//Desired(x, y, c) = fg0.values[c];// + fg1.values[c]) / 2;
		//	}
		//}
		//else
		//{
		//	Depth(x, y) = db;
		//	BestCost(x, y) = costBg;
		//	for (int c = 0; c < 3; c++)
		//	{
		//		Desired(x, y, c) = ColorBg(x,y,c);
		//		//Desired(x, y, c) = fg0.values[c];// + fg1.values[c]) / 2;
		//	}
		//	
		//}
		//else if (costBg < cost_thr)
		//{
		//	//Depth(x, y) = zb;
		//	Depth(x, y) = db;
		//	BestCost(x, y) = costBg;
		//	for (int c = 0; c < 3; c++)
		//	{
		//		Desired(x, y, c) = (bg0.values[c] + bg1.values[c]) / 2;
		//		//Desired(x, y, c) = bg0.values[c];// + bg1.values[c]) / 2;
		//	}
		//}
		//else
		//{			
		//	Depth(x, y) = nan;
		//	BestCost(x, y) = nan;
		//	for (int c = 0; c < 3; c++)
		//	{
		//		Desired(x, y, c) = nan;
		//	}
		//}

	}


	//clean-up camera images array
#pragma omp parallel for
	for (int i = 0; i<cameras; i++)
	{
		delete Images[i];
	}
}
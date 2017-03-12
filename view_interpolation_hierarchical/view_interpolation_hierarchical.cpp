/** Version of view_interpolation method, where simple hierarchical aggregation is used
* @file view_interpolation_hierarchical
* @author Sergey Smirnov
* @date 28.08.2014
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
#define cost_thr 15.f
//#define DEBUG
#define MAXDIFF 255*3+1
float getWeight2(MexImage<float> &Color1, MexImage<float> &Color2, const int x1, const int y1, const int x2, const int y2, const float * const weights_lookup)
{
	float diff1 = MAXDIFF;
	if (!isnan(Color1(x1, y1)) && !isnan(Color1(x2, y2)))
	{
		diff1 = 0;
		for (int c = 0; c < Color1.layers; c++)
		{
			diff1 += (abs(Color1(x1, y1, c) - Color1(x2, y2, c)));
		}
	}

	float diff2 = MAXDIFF;
	if (!isnan(Color2(x1, y1)) && !isnan(Color2(x2, y2)))
	{
		diff2 = 0;
		for (int c = 0; c < Color1.layers; c++)
		{
			diff2 += (abs(Color2(x1, y1, c) - Color2(x2, y2, c)));
		}
	}
	
	return diff1 < diff2 ? weights_lookup[unsigned(round(diff1))] : weights_lookup[unsigned(round(diff2))];
}

//void aggregateCost(MexImage<float> &Cost, MexImage<float> &Aggregated, MexImage<float> &Weights, MexImage<float> &Color1, MexImage<float> &Color2, float const * const weights_lookup, const int radius = 6)
//{
//	const int width = Cost.width;
//	const int height = Cost.height;
//	const long HW = Cost.layer_size;
//	Weights.setval(1.f);
//#pragma omp parallel for
//	for (long i = 0; i < HW; i++)
//	{
//		Aggregated[i] = Cost[i];
//	}
//
//#pragma omp parallel for
//	for (long i = 0; i < HW; i++)
//	{
//		const int x = i / height;
//		const int y = i % height;
//
//		float weights = 0;
//		for (int dx = -radius; dx <= radius; dx++)
//		{
//			const int xs = x + dx;
//			if (xs < 0 || xs >= width)
//			{
//				continue;
//			}
//			for (int dy = -radius; dy <= 0; dy++)
//			{
//				const int ys = y + dy;
//				if (ys < 0 || ys >= height)
//				{
//					continue;
//				}
//				const float weight = getWeight(Color1, Color2, x, y, xs, ys, weights_lookup);
//				Cost(x, y) += Aggregated(xs, ys) * weight;
//				Cost(xs, ys) += Aggregated(x, y) * weight;
//				Weights(x, y) += weight;
//				Weights(xs, ys) += weight;
//				//weights += weight;
//			}
//		}
//		//Cost(x, y) /= weights;
//	}
//#pragma omp parallel for
//	for (long i = 0; i < HW; i++)
//	{
//		Cost[i] /= Weights[i];
//	}
//
//}
//
//void aggregateCost(MexImage<float> &Cost, MexImage<float> &Buffer, MexImage<float> &Weights, MexImage<float> &Color1, MexImage<float> &Color2, float const * const weights_lookup, const float sigma_spatial = 0.1f)
//{
//	const int width = Cost.width;
//	const int height = Cost.height;
//	const long HW = Cost.layer_size;
//
//	float alpha = exp(-sqrt(2.0) / (sigma_spatial*height));
//	float inv_alpha = 1 - alpha;
//
//#pragma omp parallel for
//	for (int x = 0; x<width; x++)
//	{
//		float accumulated = Cost(x, 0);
//		Buffer(x, 0) = accumulated;
//
//		for (int y = 1; y<height; y++)
//		{
//			const float weight = getWeight(Color1, Color2, x, y - 1, x, y, weights_lookup);
//			accumulated = accumulated * weight*alpha + Cost(x, y) * inv_alpha;
//			Buffer(x, y) = accumulated;
//		}
//
//		accumulated = 0.5 * (Cost(x, height - 1) + Buffer(x, height - 1));
//		Buffer(x, height - 1) = accumulated;
//
//		for (int y = height - 2; y >= 0; y--)
//		{
//			const float weight = getWeight(Color1, Color2, x, y + 1, x, y, weights_lookup);
//			accumulated = accumulated * weight*alpha + Cost(x, y) * inv_alpha;
//			Buffer(x, y) =  0.5 * (Buffer(x, y) + accumulated);
//		}
//	}
//
//	alpha = exp(-sqrt(2.0) / (sigma_spatial*width));
//	inv_alpha = 1 - alpha;
//
//#pragma omp parallel for
//	for (int y = 0; y<height; y++)
//	{
//		float accumulated = Buffer(0, y);
//		Cost(0, y) = accumulated;
//
//		for (int x = 1; x<width; x++)
//		{
//			const float weight = getWeight(Color1, Color2, x - 1, y, x, y, weights_lookup);
//			accumulated = accumulated * weight*alpha + Buffer(x, y) * inv_alpha;
//			Cost(x, y) = accumulated;
//		}
//
//		accumulated = 0.5 * (Buffer(width - 1, y) + Cost(width - 1, y));
//		Cost(width - 1, y) = accumulated;
//
//		for (int x = width - 2; x >= 0; x--)
//		{
//			const float weight = getWeight(Color1, Color2, x + 1, y, x, y, weights_lookup);
//			accumulated = accumulated * weight*alpha + Buffer(x, y) * inv_alpha;
//			Cost(x, y) = 0.5 * (Cost(x, y) + accumulated);
//		}
//	}
//
//}
//
//#endif


float getWeight(MexImage<float> &Color1, const int x1, const int y1, const int x2, const int y2, const float * const weights_lookup)
{
	float diff1 = MAXDIFF;
	if (!isnan(Color1(x1, y1)) && !isnan(Color1(x2, y2)))
	{
		diff1 = 0;
		for (int c = 0; c < Color1.layers; c++)
		{
			diff1 += (abs(Color1(x1, y1, c) - Color1(x2, y2, c)));
		}
	}
	return weights_lookup[unsigned(round(diff1))];
}

void aggregateCost(MexImage<float> &Cost, MexImage<float> &Aggregated,  MexImage<float> &Weights, MexImage<float> &Color1, MexImage<float> &Color2, float const * const weights_lookup, float const * const  distance_lookup, const int radius)
{
	const int width = Cost.width;
	const int height = Cost.height;
	const int layers = Cost.layers;
	const long HW = Cost.layer_size;
	//const int radius = sigma_spatial*2;
	const int diameter = radius*2 + 1;
	const int window = diameter * diameter;

	for(long i=0; i<HW*layers; i++)
	{
		Aggregated[i] = Cost[i];		
	}

	for(long i=0; i<HW; i++)
	{
		Weights[i] = 1;
	}

	for(long i=0; i<HW; i++)
	{
		const int x = i / height;
		const int y = i % height;

		for(int k=0; k<window/2; k++)
		{
			const int dx = k / diameter - radius;
			const int dy = k % diameter - radius;

			const int xx = x + dx;
			const int yy = y + dy;
			if(xx<0 || xx>=width || yy<0 || yy>=height)
			{
				continue;
			}
			
			//const float weight = getWeight(Color1, x, y, xx, yy, weights_lookup) * distance_lookup[k];
			const float weight = getWeight2(Color1, Color2, x, y, xx, yy, weights_lookup) * distance_lookup[k];			
			Aggregated(x,y) += Cost(xx,yy) * weight;
			Aggregated(xx,yy) += Cost(x,y) * weight;
			Weights(x,y) += weight;
			Weights(xx,yy) += weight;			
		}
	}

	for(long i=0; i<HW; i++)
	{
		Aggregated[i] /= Weights[i];
	}
}

void projectImage(MexImage<float> &Desired, const glm::mat4 Cdinv, MexImage<float> &Image, const glm::mat4x3 Ci, const float z)
{
	const int d_width = Desired.width;
	const int d_height = Desired.height;
	const int i_width = Image.width;
	const int i_height = Image.height;

	//const long HW = width * height;

	for(int ud=0; ud<d_width; ud++)
	{
		for(int vd=0; vd<d_height; vd++)
		{
			long indexd = Image.Index(ud, vd);
			const glm::vec3 uvzi = Ci * (Cdinv * glm::vec4(ud*z, vd*z, z, 1));				

			const float ui = uvzi.x / uvzi.z;
			const float vi = uvzi.y / uvzi.z;
			const int ul = std::floor(ui);
			const int vl = std::floor(vi);

			if(ul<0 || ul >= i_width-1 || vl < 0 || vl >= i_height-1)
			{
				continue;
			}
				
			const float dx = ui-ul;
			const float dy = vi-vl;

			for(int c=0; c<colors; c++)
			{
				Desired(ud, vd, c) = Image(ul, vl, c)*(1-dx)*(1-dy) + Image(ul+1, vl, c)*dx*(1-dy) + Image(ul,vl+1,c)*(1-dx)*dy + Image(ul+1,vl+1,c)*dx*dy;													
			}			
		}
	}
}

void mexFunction (int nout, mxArray* output[], int in, const mxArray* input[])
{
	#ifndef _DEBUG
	omp_set_num_threads(std::max(4,omp_get_max_threads())); 
	omp_set_dynamic(std::max(2, omp_get_max_threads() / 2));
	#endif

	const int _offset = 6; // 5 obligatory params goes before variable numner of camera pairs

	if (in < _offset + 4 || (in - _offset) % 4 != 0 || nout != 3 || mxGetClassID(input[0]) != mxSINGLE_CLASS)
	{
		mexPrintf("View interpolation with Bilateral Cost Aggregation\n");
		mexErrMsgTxt("USAGE: [Desired, Depth, BestCost] = view_interpolation_hierarchical(C_desired, minZ, maxZ, layers, pyramid_levels, sigma_color, single(Image1), C1, single(Image2), C2, <,..., ImageN, CN>);");
	}

	
	const float minZ = static_cast<float>(mxGetScalar(input[1]));	
	const float maxZ = static_cast<float>(mxGetScalar(input[2]));	
	const int layers = static_cast<int>(mxGetScalar(input[3]));	
	const unsigned levels = static_cast<unsigned>(mxGetScalar(input[4]));
	const float sigma_color = static_cast<float>(mxGetScalar(input[5]));
	std::auto_ptr<float> weights_lookup(new float[256 * colors]);
	for (int i = 0; i < 256 * colors; i++)
	{
		weights_lookup.get()[i] = exp(-float(i) / (colors * sigma_color));
	}
	
		
	const int cameras = (in-_offset)/2;
	const int height = mxGetDimensions(input[_offset])[0];		
	const int width = mxGetDimensions(input[_offset])[1];
	//const int colors = mxGetDimensions(input[_offset])[2];
	const long HW = width*height;

	if(cameras < 2)
	{
		mexErrMsgTxt("At least two camera frames must be provided!"); 
	}

	if (cameras % 2)
	{
		mexErrMsgTxt("There must be even number of cameras! Each 2 consequtive cameras are treated as stereo pair!");
	}

	if((mxGetDimensions(input[0]))[0] != 3 || (mxGetDimensions(input[0]))[1] != 4)
	{
		mexErrMsgTxt("C_desired matrix must be 4x3"); 
	}

	const float * const cd = (float*) mxGetData(input[0]);
	const glm::mat4x3 Cd = glm::mat4x3(cd[0],cd[1],cd[2],cd[3],cd[4],cd[5],cd[6],cd[7],cd[8],cd[9],cd[10],cd[11]);
	const glm::mat4 Cdinv = glm::inverse(glm::mat4(Cd));
	//const float cost_thr = 40.f;

	// double-check input data
	for(int n=0; n<cameras; n++)
	{
		if(mxGetClassID(input[_offset+n*2]) != mxSINGLE_CLASS)
		{
			char buff[1000];
			sprintf(buff, "Camera frame %d must be of a SINGLE-type ", n+1);
			mexErrMsgTxt(buff);
		}
		
		if((mxGetDimensions(input[_offset+n*2]))[0] != height || (mxGetDimensions(input[_offset+n*2]))[1] != width || (mxGetDimensions(input[_offset+n*2]))[2] != colors)
		{			
			mexErrMsgTxt("All camera frames must have the same resolution and three colors!");
		}

		if(mxGetClassID(input[_offset+n*2+1])!=mxSINGLE_CLASS)
		{
			char buff[1000];
			sprintf(buff, "Camera matrix %d must be of a SINGLE-type ", n+1);
			mexErrMsgTxt(buff);
		}

		if((mxGetDimensions(input[_offset+n*2+1]))[0] != 3 || (mxGetDimensions(input[_offset+n*2+1]))[1] != 4)
		{
			char buff[1000];
			sprintf(buff, "Camera matrix %d must be of sized 3x4", n+1);
			mexErrMsgTxt(buff);
		}
	}
	
	MexImage<float>** Images = new MexImage<float>*[cameras];

	glm::mat4x3 *C = new glm::mat4x3[cameras];
	//glm::mat4 *Cinv = new glm::mat4[cameras];
	std::auto_ptr<float> distances(new float[cameras]);
	
	#pragma omp parallel for
	for(int n=0; n<cameras; n++)
	{		
		Images[n] = new MexImage<float>(input[_offset + n*2]);
		//Colors[n] = new MexImage<float>(width, height, colors);

		const float * const c1 = (float*)mxGetData(input[_offset + n*2 + 1]);
		C[n] = glm::mat4x3(c1[0],c1[1],c1[2],c1[3],c1[4],c1[5],c1[6],c1[7],c1[8],c1[9],c1[10],c1[11]);	
		//Cinv[n] = glm::inverse(glm::mat4(C[n]));
		distances.get()[n] = sqrt((c1[9] - cd[9])*(c1[9] - cd[9]) + (c1[10] - cd[10])*(c1[10] - cd[10]) + (c1[11] - cd[11])*(c1[11] - cd[11]));
	}			

	const float nan = sqrt(-1.f);

	const mwSize dimC[] = {(unsigned)height, (unsigned)width, (unsigned)colors};
	const mwSize dims[] = {(unsigned)height, (unsigned)width, 1};
	
	//Matlab-allocated variables
	output[0] = mxCreateNumericArray(3, dimC, mxSINGLE_CLASS, mxREAL);
	output[1] = mxCreateNumericArray(3, dims, mxSINGLE_CLASS, mxREAL);
	output[2] = mxCreateNumericArray(3, dims, mxSINGLE_CLASS, mxREAL);

	MexImage<float> Desired(output[0]); // desired color view
	MexImage<float> Depth(output[1]); // estimated depth in the desired view
	MexImage<float> BestCost(output[2]); // estimated best cost
	
	BestCost.setval(cost_thr);


	// for each depth-layer in the space
	#pragma omp parallel 
	{
		MexImage<float> Cost(width, height);
		MexImage<float> Weights(width, height);
		MexImage<float> Aggregated(width, height);


		#pragma omp for		
		for (int d = 0; d < layers; d++)
		{
			float z = 1.f / ((float(d) / layers)*(1.f / minZ - 1.f / maxZ) + 1.f / maxZ);

			MexImage<float> Color1(width, height, colors);
			MexImage<float> Color2(width, height, colors);

			for (int n = 0; n < cameras; n += 2)
			{
				Cost.setval(0.f);
				Aggregated.setval(0.f);
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
						Cost[i] = cost_thr;
						continue;
					}
					float diff = 0.f;
					for (int c = 0; c < colors; c++)
					{
						diff += abs(Color1[i + c*HW] - Color2[i + c*HW]);
					}
					diff /= colors;
					diff = diff > cost_thr ? cost_thr : diff;

					Cost[i] += diff; // addition is needed when we have more than one pair
					//Cost[i] = diff;
				}

				//aggregateCost(Cost, Aggregated, Color1, weights_lookup.get(), sigma_spatial);
				aggregateCost(Cost, Aggregated, Weights, Color1, Color2, weights_lookup.get(), distance_lookup.get(), radius);
				
				const float w1 = 1-distances.get()[n] / (distances.get()[n] + distances.get()[n + 1]);
				const float w2 = 1-distances.get()[n+1] / (distances.get()[n] + distances.get()[n + 1]);
				
				#pragma omp parallel for
				for (long i = 0; i < HW; i++)
				{
					const int x = i / height;
					const int y = i % height;

					const float aggr_cost = Aggregated[i];

					if (aggr_cost < BestCost(x, y))
					{
						#pragma omp critical
						{
							BestCost(x, y) = aggr_cost;
							Depth(x, y) = d;
							if (isnan(Color1(x, y, 0)) && !isnan(Color2(x, y, 0)))
							{
								for (int c = 0; c < colors; c++)
								{
									Desired(x, y, c) = Color2(x, y, c);
								}
							}
							else if (!isnan(Color1(x, y, 0)) && isnan(Color2(x, y, 0)))
							{
								for (int c = 0; c < colors; c++)
								{
									Desired(x, y, c) = Color1(x, y, c);
								}
							}
							else
							{
								for (int c = 0; c < colors; c++)
								{
									Desired(x, y, c) = Color1(x, y, c)*w1 + Color2(x, y, c)*w2;
								}
							}							
						}
					}
				}
			}
		}
	}		
	

	//clean-up camera images array
	#pragma omp parallel for
	for(int i=0; i<cameras; i++)
	{
		delete Images[i];
	}
	
	delete[] Images;

	// delete camera matrix array
	delete[] C;

}
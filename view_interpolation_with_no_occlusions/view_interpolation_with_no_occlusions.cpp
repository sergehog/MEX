/** Interpolates (synthesizes) desired view between number of given rectified camera images. Uses special treatment for dis-occluded areas
* @file view_interpolation_with_no_occlusions.cpp
* @author Sergey Smirnov <sergey.smirnov@tut.fi>
* @date 20.05.2014
* @copyright 3D Media Group / Tampere University of Technology
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


float getWeight(const MexImage<float> &Color, const int x1, const int y1, const int x2, const int y2, const float * const weights_lookup)
{
	float diff = 0;
	if (!isnan(Color(x1, y1)) && !isnan(Color(x2, y2)))
	{
		for (int c = 0; c < Color.layers; c++)
		{
			diff += (abs(Color(x1, y1, c) - Color(x2, y2, c)));
		}
	}
	else
	{
		return 0;
	}

	return weights_lookup[unsigned(round(diff))];		
}

void recursive_bilateral(MexImage<float> &Signal, MexImage<float> &Temporal, MexImage<float> &Weights, MexImage<float> &Weights2, const MexImage<float> &Color, float const * const weights_lookup, float const sigma_spatial)
{	
	const int width = Signal.width;
	const int height = Signal.height;	
	const long HW = Signal.layer_size;	

	//const float alpha = exp(-sqrt(2.0) / (sigma_spatial*std::min(height, width)));
	const float alpha = sigma_spatial;

	Weights.setval(0.f);	
	Temporal.setval(0.f);	


	// horisontal passes
	#pragma omp parallel for
	for(int y=0; y<height; y++)
	{
		float t1 = Signal(0,y);
		float t2 = Signal(width-1,y);				
		float w1 = 1.f;
		float w2 = 1.f;			

		Temporal(0,y) = t1;
		Temporal(width-1,y) = t2;
		Weights(0, y) = w1;
		Weights(width-1, y) = w2;

		for(int x1=1; x1<width; x1++)
		{
			#pragma omp parallel sections
			{
				#pragma omp section
				{					
					const float weight1 = getWeight(Color, x1, y, x1-1,y,weights_lookup);
					t1 = Signal(x1,y) + t1 * alpha*weight1;
					w1 = (1 + w1 * alpha*weight1);
					#pragma omp atomic
					Temporal(x1,y) += t1;
					#pragma omp atomic
					Weights(x1, y) += w1;
				}
				#pragma omp section
				{
					const int x2 = width - x1 - 1;
					const float weight2 = getWeight(Color, x2, y, x2+1,y,weights_lookup);
					t2 = Signal(x2,y) + t2 * alpha*weight2;							
					w2 = (1 + w2 * alpha*weight2);			
					#pragma omp atomic
					Temporal(x2,y) += t2;			
					#pragma omp atomic
					Weights(x2, y) += w2;
				}
			}			
		}

		#pragma omp parallel for
		for(int x=0; x<width; x++)
		{				
			Weights(x, y) -= 1;
			Temporal(x,y) -= Signal(x,y);
		}
	}	
	
	Signal.setval(0.f);
	Weights2.setval(0.f);

	//vertical passes		
	#pragma omp parallel for
	for(int x=0; x<width; x++)
	{
		float t1 = Temporal(x,0);
		float t2 = Temporal(x, height-1);				
		float w1 = Weights(x, 0);
		float w2 = Weights(x, height-1);		

		Signal(x,0) = t1;
		Signal(x, height-1) = t2;
		Weights2(x,0) = w1;
		Weights2(x, height-1) = w2;

		for(int y1=1; y1<height; y1++)
		{
			#pragma omp parallel sections
			{
				#pragma omp section
				{	
					const float weight1 = getWeight(Color, x, y1, x,y1-1,weights_lookup);
					t1 = Temporal(x,y1) + t1 * alpha*weight1;
					w1 = (Weights(x, y1)  + w1 * alpha*weight1);
					#pragma omp atomic
					Signal(x,y1) += t1;
					#pragma omp atomic
					Weights2(x,y1) += w1;
				}
				#pragma omp section
				{
					const int y2 = height - y1 - 1;
					const float weight2 = getWeight(Color, x, y2, x,y2+1,weights_lookup);
					t2 = Temporal(x, y2) + t2 * alpha*weight2;							
					w2 = (Weights(x, y2)  + w2 * alpha*weight2);			
					#pragma omp atomic
					Signal(x, y2) += t2;			
					#pragma omp atomic
					Weights2(x, y2) += w2;
				}
			}
		}

		#pragma omp parallel for
		for(int y=0; y<height; y++)
		{				
			Weights2(x,y) -= Weights(x, y);
			Signal(x,y) -= Temporal(x,y);
		}
	}
	
	// final normalization
	#pragma omp parallel for
	for(long i=0; i<HW; i++)
	{
		Signal[i] /= Weights2[i];
	}
}

#define LIM(A,B) ((A) < 0 ? 0 : ((A) > (B)-1 ? (B)-1 : (A)))

void projectImage_CLAMP_TO_EDGE(MexImage<float> &Desired, const glm::mat4 Cdinv, MexImage<float> &Image, const glm::mat4x3 Ci, const float z)
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
						
			const float dx = ui-ul;
			const float dy = vi-vl;

			//if(ul<0 || ul >= i_width-1 || vl < 0 || vl >= i_height-1)
			//{
			//	continue;
			//}
				
			#pragma loop(hint_parallel(colors))
			for(int c=0; c<colors; c++)
			{
				Desired(ud, vd, c) = Image(LIM(ul,i_width), LIM(vl,i_height), c)*(1-dx)*(1-dy) 
					+ Image(LIM(ul+1,i_width), LIM(vl,i_height), c)*dx*(1-dy) 
					+ Image(LIM(ul,i_width),LIM(vl+1,i_height),c)*(1-dx)*dy 
					+ Image(LIM(ul+1,i_width),LIM(vl+1,i_height),c)*dx*dy;													
			}			
		}
	}
}

#define SUMCOST // summate cost values, otherwise choose minimums

void mexFunction (int nout, mxArray* output[], int in, const mxArray* input[])
{
	#ifndef _DEBUG
	omp_set_num_threads(std::max(4,omp_get_max_threads())); 
	omp_set_dynamic(std::max(2, omp_get_max_threads() / 2));
	#endif

	const int _offset = 6; // 5 obligatory params goes before variable numner of camera pairs

	//if (in < _offset + 4 || (in - _offset) % 4 != 0 || nout != 3 || mxGetClassID(input[0]) != mxSINGLE_CLASS)
	if (in < _offset + 4 || nout != 4 || mxGetClassID(input[0]) != mxSINGLE_CLASS)
	{
		mexPrintf("View interpolation with Compensated dis-occlusions\n");
		mexErrMsgTxt("USAGE: [Desired, Depth, BestCost, Weights] = view_interpolation_with_no_occlusions(C_desired, minZ, maxZ, layers, sigma_color, sigma_spatial, single(Image1), C1, single(Image2), C2, <,..., ImageN, CN>);");
	}
	
	const float minZ = static_cast<float>(mxGetScalar(input[1]));	
	const float maxZ = static_cast<float>(mxGetScalar(input[2]));	
	const int layers = static_cast<int>(mxGetScalar(input[3]));	

	const float sigma_color = std::max(0.f,std::min(1.f, static_cast<float>(mxGetScalar(input[4]))));
	const float sigma_spatial = std::max(0.f,std::min(1.f, static_cast<float>(mxGetScalar(input[5]))));	

	std::unique_ptr<float[]> weights_lookup(new float[256 * colors]);
	for (int i = 0; i < 256 * colors; i++)
	{
		weights_lookup[i] = exp(-float(i) / (255 * colors * sigma_color));
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

	//if (cameras % 2)
	//{
	//	mexErrMsgTxt("There must be even number of cameras! Each 2 consequtive cameras are treated as stereo pair!");
	//}

	if((mxGetDimensions(input[0]))[0] != 3 || (mxGetDimensions(input[0]))[1] != 4)
	{
		mexErrMsgTxt("C_desired matrix must be 4x3"); 
	}
		
	const float mu_regularization = 0.1;
	const float * const cd = (float*) mxGetData(input[0]);
	const glm::mat4x3 Cd = glm::mat4x3(cd[0],cd[1],cd[2],cd[3],cd[4],cd[5],cd[6],cd[7],cd[8],cd[9],cd[10],cd[11]);
	const glm::mat4 Cdinv = glm::inverse(glm::mat4(Cd));
	//const float cost_thr = 40.f;

	//! check input data
	for(int n=0; n<cameras; n++)
	{
		if(mxGetClassID(input[_offset+n*2]) != mxSINGLE_CLASS)
		{
			mexPrintf("Camera image %d must be of a SINGLE-type", n+1);
			mexErrMsgTxt("");			
		}
		
		if((mxGetDimensions(input[_offset+n*2]))[0] != height || (mxGetDimensions(input[_offset+n*2]))[1] != width || (mxGetDimensions(input[_offset+n*2]))[2] != colors)
		{			
			mexErrMsgTxt("All camera frames must have the same resolution and three colors!");
		}

		if(mxGetClassID(input[_offset+n*2+1])!=mxSINGLE_CLASS)
		{
			mexPrintf("Camera matrix %d must be of a SINGLE-type", n+1);
			mexErrMsgTxt("");
		}

		if((mxGetDimensions(input[_offset+n*2+1]))[0] != 3 || (mxGetDimensions(input[_offset+n*2+1]))[1] != 4)
		{
			mexPrintf("Camera matrix %d must be of size 3x4", n+1);
			mexErrMsgTxt("");
		}
	}
	
	MexImage<float>** Images = new MexImage<float>*[cameras];

	glm::mat4x3 *C = new glm::mat4x3[cameras];
	//glm::mat4 *Cinv = new glm::mat4[cameras];
	std::unique_ptr<float[]> distances(new float[cameras]);
	
	#pragma omp parallel for
	for(int n=0; n<cameras; n++)
	{		
		Images[n] = new MexImage<float>(input[_offset + n*2]);
		//Colors[n] = new MexImage<float>(width, height, colors);

		const float * const c1 = (float*)mxGetData(input[_offset + n*2 + 1]);
		C[n] = glm::mat4x3(c1[0],c1[1],c1[2],c1[3],c1[4],c1[5],c1[6],c1[7],c1[8],c1[9],c1[10],c1[11]);	
		//Cinv[n] = glm::inverse(glm::mat4(C[n]));
		distances[n] = sqrt((c1[9] - cd[9])*(c1[9] - cd[9]) + (c1[10] - cd[10])*(c1[10] - cd[10]) + (c1[11] - cd[11])*(c1[11] - cd[11]));
	}			

	const float nan = sqrt(-1.f);

	const mwSize dimC[] = {(unsigned)height, (unsigned)width, (unsigned)colors};
	const mwSize dims[] = {(unsigned)height, (unsigned)width, 1};
	const mwSize dimsK[] = {(unsigned)height, (unsigned)width, cameras};
	
	//Matlab-allocated variables
	output[0] = mxCreateNumericArray(3, dimC, mxSINGLE_CLASS, mxREAL);
	output[1] = mxCreateNumericArray(3, dims, mxSINGLE_CLASS, mxREAL);
	output[2] = mxCreateNumericArray(3, dims, mxSINGLE_CLASS, mxREAL);
	output[3] = mxCreateNumericArray(3, dimsK, mxSINGLE_CLASS, mxREAL);

	MexImage<float> Desired(output[0]); // desired color view
	MexImage<float> Depth(output[1]); // estimated depth in the desired view
	MexImage<float> BestCost(output[2]); // estimated best cost
	MexImage<float> AveragingWeights(output[3]); // estimated best cost
	
	BestCost.setval(cost_thr);


	// for each depth-layer in the space
#pragma omp parallel 
	{
		//MexImage<float> Cost(width, height);
		MexImage<float> Weights(width, height);
		MexImage<float> Weights2(width, height);
		MexImage<float> Temporal(width, height);
		//MexImage<float> Aggregated2(width, height);

		MexImage<float> **Colors = new MexImage<float>*[cameras];
		MexImage<float> **Costs = new MexImage<float>*[cameras];
		for(int n=0; n<cameras; n++)
		{
			Colors[n] = new MexImage<float>(width, height, colors);
			Costs[n] = new MexImage<float>(width, height);
		}
		

		#pragma omp for		
		for (int d = 0; d < layers; d++)
		{
			const float z = 1.f / ((float(d) / layers)*(1.f / minZ - 1.f / maxZ) + 1.f / maxZ);

			// project all images to a position of a desired camera O(N*L) complexity
			#pragma omp parallel for
			for (int n = 0; n < cameras; n ++)
			{
				MexImage<float> &Color = *Colors[n];
				MexImage<float> &Cost = *Costs[n];
				MexImage<float> &Image = *Images[n];
				Color.setval(nan);
				projectImage_CLAMP_TO_EDGE(Color, Cdinv, Image, C[n], z);
#ifdef SUMCOST
				Cost.setval(0.f);
#elif 
				Cost.setval(cost_thr);
#endif								
			}

			// O(N*L) complexity part (only pairs, specified in the input)
			#pragma omp parallel for
			for (int n = 0; n < cameras; n +=2)			
			{
				MexImage<float> &Color1 = *Colors[n];
				MexImage<float> &Cost1 = *Costs[n];
				MexImage<float> &Color2 = *Colors[n+1];
				MexImage<float> &Cost2 = *Costs[n+1];
				for (long i = 0; i < HW; i++)
				{
					float value = 0;
					if (isnan(Color1[i]) || isnan(Color2[i]))
					{
						value = cost_thr;
					}
					else
					{
						for (int c = 0; c < colors; c++)
						{
							value += abs(Color1[i + c*HW] - Color2[i + c*HW]);
						}
						value /= colors;
						value = value > cost_thr ? cost_thr : value;
					}
						
#ifdef SUMCOST					
					Cost1[i] += (!isnan(Color1[i])) ? value : cost_thr * 2; 
					Cost2[i] += (!isnan(Color2[i])) ? value : cost_thr * 2; 
#elif 
					Cost1[i] = std::min(value, Cost1[i]);
					Cost2[i] = std::min(value, Cost2[i]);
#endif								
				}
			}

			//// O(N^2*L) complexity version (all stereo pairs)
			//#pragma omp parallel for
			//for (int n1 = 0; n1 < cameras; n1 ++)			
			//{
			//	MexImage<float> &Color1 = *Colors[n1];
			//	MexImage<float> &Cost1 = *Costs[n1];

			//	for (int n2 = n1+1 ; n2 < cameras; n2 ++)
			//	{					
			//		MexImage<float> &Color2 = *Colors[n2];
			//		MexImage<float> &Cost2 = *Costs[n2];

			//		for (long i = 0; i < HW; i++)
			//		{
			//			float value = 0;
			//			if (isnan(Color1[i]) || isnan(Color2[i]))
			//			{
			//				value = cost_thr;
			//			}
			//			else
			//			{
			//				for (int c = 0; c < colors; c++)
			//				{
			//					value += abs(Color1[i + c*HW] - Color2[i + c*HW]);
			//				}
			//				value /= colors;
			//				value = value > cost_thr ? cost_thr : value;
			//			}
			//			
			//			Cost1[i] += value/cameras;
			//			Cost2[i] += value/cameras;
			//			//Cost1[i] = std::min(value, Cost1[i]);
			//			//Cost2[i] = std::min(value, Cost2[i]);
			//		}
			//	}
			//}
			
			// aggregation O(N*L) complexity
			#pragma omp parallel for
			for (int n = 0; n < cameras; n ++)
			{
				MexImage<float> &Cost = *Costs[n];
				MexImage<float> &Color = *Colors[n];
				recursive_bilateral(Cost, Temporal, Weights, Weights2, Color, weights_lookup.get(), sigma_spatial);				
			}
			
			#pragma omp parallel for
			for(long i=0; i<HW; i++)
			{
				const int x = i / height;
				const int y = i % height;
				
				for(int n=0; n<cameras; n++)
				{
					MexImage<float> &Cost = *Costs[n];					
					if(Cost[i] < BestCost[i])
					{
						float weights = 0;
						BestCost[i] = Cost[i];
						Depth[i] = z;//d;						

						for(int c=0; c<colors; c++)
						{
							Desired(x, y, c) = 0;
						}

						for(int nx=0; nx<cameras; nx++)
						{							
							MexImage<float> &CostX = *Costs[nx];
							MexImage<float> &ColorX = *Colors[nx];
							AveragingWeights(x, y, nx) = CostX(x,y);

							float w = cost_thr/(CostX[i] < mu_regularization ? mu_regularization : CostX[i]);							
							
							//float w = 1;
							
							if(!isnan(ColorX(x, y, 0)))
							{
								weights += w;  
								for(int c=0; c<colors; c++)
								{								
									Desired(x, y, c) += ColorX(x, y, c) * w;
								}
							}
						}
						
						for(int c=0; c<colors; c++)
						{								
							Desired(x, y, c) /= weights;
						}

						//break;
					}
				}
			}
			
		}

		for(int n=0; n<cameras; n++)
		{
			delete Colors[n];
			delete Costs[n];
		}

		delete[] Colors;
		delete[] Costs;
	}		
	
	
	//clean-up camera images array
#pragma omp parallel for
	for(int i=0; i<cameras; i++)
	{
		delete Images[i];
	}
	
	delete[] Images;// , Colors;

	// delete camera matrix array
	delete[] C;//, Cinv;

}
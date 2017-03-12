/** Interpolates desired view between (number of) given calibrated camera images
* @file view_interpolation.cpp
* @author Sergey Smirnov <sergey.smirnov@tut.fi> / 3D Media Group / Tampere University of Technology
* @date 18.12.2013
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
float getWeight2(const MexImage<float> &Color1, const MexImage<float> &Color2, const int x1, const int y1, const int x2, const int y2, const float * const weights_lookup)
{
	float diff1 = 0;
	if (!isnan(Color1(x1, y1)) && !isnan(Color1(x2, y2)))
	{
		//diff1 = 0;
		for (int c = 0; c < Color1.layers; c++)
		{
			diff1 += (abs(Color1(x1, y1, c) - Color1(x2, y2, c)));
		}
	}
	else
	{
		diff1 = MAXDIFF;
	}

	//return weights_lookup[unsigned(round(diff1))];

	float diff2 = 0;
	if (!isnan(Color2(x1, y1)) && !isnan(Color2(x2, y2)))
	{
		//diff2 = 0;
		for (int c = 0; c < Color1.layers; c++)
		{
			diff2 += (abs(Color2(x1, y1, c) - Color2(x2, y2, c)));
		}
	}
	else
	{
		diff2 = MAXDIFF;
	}
	
	//return weights_lookup[unsigned(round(diff2))];

	return diff1 < diff2 ? weights_lookup[unsigned(round(diff1))] : weights_lookup[unsigned(round(diff2))];
	
}

void recursive_bilateral(MexImage<float> &Signal, MexImage<float> &Temporal, MexImage<float> &Weights, MexImage<float> &Weights2, const MexImage<float> &Color1, const MexImage<float> &Color2, float const * const weights_lookup, float const sigma_spatial)
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
					const float weight1 = getWeight2(Color1, Color2, x1, y, x1-1,y,weights_lookup);
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
					const float weight2 = getWeight2(Color1, Color2, x2, y, x2+1,y,weights_lookup);
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
					const float weight1 = getWeight2(Color1, Color2, x, y1, x,y1-1,weights_lookup);
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
					const float weight2 = getWeight2(Color1, Color2, x, y2, x,y2+1,weights_lookup);
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

			float ui = uvzi.x / uvzi.z;
			float vi = uvzi.y / uvzi.z;
			ui = ui < 0 ? 0 : ui >= i_width-1 ? i_width-1 : ui;
			vi = vi < 0 ? 0 : vi >= i_height-1 ? i_height-1 : vi;
			
			const int ul = std::floor(ui);
			const int vl = std::floor(vi);
			
			//if(ul<0 || ul >= i_width-1 || vl < 0 || vl >= i_height-1)
			//{
			//	continue;
			//}
				
			const float dx = ui-ul;
			const float dy = vi-vl;

			#pragma loop(hint_parallel(colors))
			for(int c=0; c<colors; c++)
			{
				Desired(ud, vd, c) = Image(ul, vl, c)*(1-dx)*(1-dy) + Image(ul+1, vl, c)*dx*(1-dy) + Image(ul,vl+1,c)*(1-dx)*dy + Image(ul+1,vl+1,c)*dx*dy;													
			}			
		}
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
		mexErrMsgTxt("USAGE: [Desired, Depth, BestCost] = view_interpolation(C_desired, minZ, maxZ, layers, sigma_color, sigma_spatial, single(Image1), C1, single(Image2), C2, <,..., ImageN, CN>);");
	}
	
	const float minZ = static_cast<float>(mxGetScalar(input[1]));	
	const float maxZ = static_cast<float>(mxGetScalar(input[2]));	
	const int layers = static_cast<int>(mxGetScalar(input[3]));	
	const float sigma_color = std::max(0.f,std::min(1.f, static_cast<float>(mxGetScalar(input[4]))));
	const float sigma_spatial = std::max(0.f,std::min(1.f, static_cast<float>(mxGetScalar(input[5]))));

	std::unique_ptr<float[]> weights_lookup(new float[256 * colors]);
	for (int i = 0; i < 256 * colors; i++)
	{
		weights_lookup[i] = exp(-float(i) / (colors * 255 * sigma_color));
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
		MexImage<float> Weights2(width, height);
		MexImage<float> Temporal(width, height);

		#pragma omp for		
		for (int d = 0; d < layers; d++)
		{
			float z = 1.f / ((float(d) / layers)*(1.f / minZ - 1.f / maxZ) + 1.f / maxZ);

			MexImage<float> Color1(width, height, colors);
			MexImage<float> Color2(width, height, colors);

			for (int n = 0; n < cameras; n += 2)
			{
				Cost.setval(0.f);
				//Aggregated.setval(0.f);
				MexImage<float> &Image1 = *Images[n];
				MexImage<float> &Image2 = *Images[n + 1];
				Color1.setval(nan);
				projectImage_CLAMP_TO_EDGE(Color1, Cdinv, Image1, C[n], z);
				Color2.setval(nan);
				projectImage_CLAMP_TO_EDGE(Color2, Cdinv, Image2, C[n + 1], z);
				
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

				
				recursive_bilateral(Cost, Temporal, Weights, Weights2, Color1, Color2, weights_lookup.get(), sigma_spatial);
				
				const float w1 = 1-distances.get()[n] / (distances.get()[n] + distances.get()[n + 1]);
				const float w2 = 1-distances.get()[n+1] / (distances.get()[n] + distances.get()[n + 1]);
				
				#pragma omp parallel for
				for (long i = 0; i < HW; i++)
				{
					const int x = i / height;
					const int y = i % height;

					const float aggr_cost = Cost[i];

					if (aggr_cost < BestCost(x, y))
					{
						#pragma omp critical
						{
							BestCost(x, y) = aggr_cost;
							Depth(x, y) = z;//d;
							if (isnan(Color1(x, y, 0)) && !isnan(Color2(x, y, 0)))
							{
								#pragma omp parallel for
								for (int c = 0; c < colors; c++)
								{
									Desired(x, y, c) = Color2(x, y, c);
								}
							}
							else if (!isnan(Color1(x, y, 0)) && isnan(Color2(x, y, 0)))
							{
								#pragma omp parallel for
								for (int c = 0; c < colors; c++)
								{
									Desired(x, y, c) = Color1(x, y, c);
								}
							}
							else
							{
								#pragma omp parallel for
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
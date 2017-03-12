/** 
* @file view_interpolation_box
* @author Sergey Smirnov
* @date 20.05.2014
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

	const int _offset = 5; // 5 obligatory params goes before variable numner of camera pairs

	//if (in < _offset + 4 || (in - _offset) % 4 != 0 || nout != 3 || mxGetClassID(input[0]) != mxSINGLE_CLASS)
	if (in < _offset + 4 || nout != 3 || mxGetClassID(input[0]) != mxSINGLE_CLASS)
	{
		mexPrintf("View interpolation with Fixed-Kernel Aggregation\n");
		mexErrMsgTxt("USAGE: [Desired, Depth, BestCost] = view_interpolation_with_no_occlusions(C_desired, minZ, maxZ, layers, radius, single(Image1), C1, single(Image2), C2, <,..., ImageN, CN>);");
	}
	
	const float minZ = static_cast<float>(mxGetScalar(input[1]));	
	const float maxZ = static_cast<float>(mxGetScalar(input[2]));	
	const int layers = static_cast<int>(mxGetScalar(input[3]));	

	const unsigned radius = static_cast<unsigned>(mxGetScalar(input[4]));	

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
	
	
	const float mu_regularization = 0.01;
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
	const int triangular_number = cameras*(cameras-1)/2;
	
	// for each depth-layer in the space
#pragma omp parallel 
	{
		MexImage<float> Cost(width, height);

		std::unique_ptr<MexImage<float> *[]> Colors = std::unique_ptr<MexImage<float> *[]>(new MexImage<float>*[cameras]);		

		for(int n=0; n<cameras; n++)
		{
			Colors[n] = new MexImage<float>(width, height, colors);
		}
		
		std::unique_ptr<float[]> updated_color = std::unique_ptr<float[]>(new float[colors]);

		#pragma omp for		
		for (int d = 0; d < layers; d++)
		{
			const float z = 1.f / ((float(d) / layers)*(1.f / minZ - 1.f / maxZ) + 1.f / maxZ);
			
			Cost.setval(0.f);
			// project all images to a position of a desired camera O(N*L) complexity
			
			for (int n = 0; n < cameras; n ++)
			{
				MexImage<float> &Color = *Colors[n];				
				MexImage<float> &Image = *Images[n];
				Color.setval(nan);
				projectImage_CLAMP_TO_EDGE(Color, Cdinv, Image, C[n], z);
			}			
			
			for (int n1 = 0; n1 < cameras; n1 ++)			
			{
				MexImage<float> &Color1 = *Colors[n1];				
				for(int n2=n1+1; n2<cameras; n2++)
				{
					
					MexImage<float> &Color2 = *Colors[n2];
					for (long i = 0; i < HW; i++)
					{
						float value = 0;
						if (isnan(Color1[i]) || isnan(Color2[i]))
						{
							value = cost_thr/triangular_number;
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
						
						Cost[i] += value/triangular_number;					
					}
				}
			}

			
			Cost.IntegralImage(true);

			for(long i=0; i<HW; i++)
			{
				const int x = i / height;
				const int y = i % height;
				float cost = Cost.getIntegralAverage(x,y,radius);
				if(cost < BestCost[i])
				{
					for(int c=0; c<colors; c++)
					{
						updated_color[c] = 0;
					}

					float weights = 0;

					for(int n=0; n<cameras; n++)
					{					
											
						MexImage<float> &Color = *Colors[n];
						const float w = 1;
						if(!isnan(Color(x, y, 0)))
						{
							weights += 1;//w;  
							for(int c=0; c<colors; c++)
							{					
								updated_color[c] += Color(x, y, c) ;
							}
						}
					}

					for(int c=0; c<colors; c++)
					{
						updated_color[c] /= weights;
					}

					while(cost < BestCost[i])
					{						
						#pragma omp critical
						{
							for(int c=0; c<colors; c++)
							{								
								Desired(x, y, c) = updated_color[c];
							}
							Depth[i] = z;
							BestCost[i] = cost;
						}
					}

				}
				
			}
			
		}

		for(int n=0; n<cameras; n++)
		{
			delete Colors[n];
		}

		delete[] Colors;
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
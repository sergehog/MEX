/** Synthesizes a novel view from view+depth frame(s). Uses layered interpolation-like method in order to avoid cracks and Reg2Non-Reg resampling
* @file depth_interpolation.cpp
* @author Sergey Smirnov <sergey.smirnov@tut.fi>
* @date 23.10.2014
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
#define cost_thr 20.f

//#define MAXDIFF 255*3+1

#define LIM(A,B) ((A) < 0 ? 0 : ((A) > (B)-1 ? (B)-1 : (A)))

void projectDepth(MexImage<float> &Desired, const glm::mat4 Cdinv, MexImage<float> &Depth, const glm::mat4x3 Ci, const float z, const float thr)
{
	Desired.setval(thr);

	const int d_width = Desired.width;
	const int d_height = Desired.height;
	const int i_width = Depth.width;
	const int i_height = Depth.height;

	//const long HW = width * height;

	for(int ud=0; ud<d_width; ud++)
	{
		for(int vd=0; vd<d_height; vd++)
		{
			long indexd = Depth.Index(ud, vd);
			const glm::vec3 uvzi = Ci * (Cdinv * glm::vec4(ud*z, vd*z, z, 1));				

			const float ui = uvzi.x / uvzi.z;
			const float vi = uvzi.y / uvzi.z;
			const int ul = std::floor(ui);
			const int vl = std::floor(vi);
						
			const float dx = ui-ul;
			const float dy = vi-vl;

			if(ul<0 || ul >= i_width-1 || vl < 0 || vl >= i_height-1)
			{
				continue;
			}

			const float z_found = Depth(ul, vl)*(1-dx)*(1-dy) + Depth(ul+1, vl)*dx*(1-dy) + Depth(ul,vl+1)*(1-dx)*dy + Depth(ul+1,vl+1)*dx*dy;
			
			Desired(ud, vd) = isnan(z_found) ? thr : abs(z_found - uvzi.z);					
		}
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

	const int _offset = 4; // obligatory params goes before variable number of cameras

	if (in < _offset + 3 || (in - _offset - 3) % 3 != 0 || nout != 2 )//|| mxGetClassID(input[4]) != mxSINGLE_CLASS || mxGetClassID(input[5]) != mxSINGLE_CLASS || mxGetClassID(input[6]) != mxSINGLE_CLASS)
	{
		mexPrintf("Depth Estimation for a reference camera\n");
		mexErrMsgTxt("USAGE: [Iv, Zv] = depth_interpolation(Cdesired, minZ, maxZ, layers, single(Image1), single(Z1), single(C1), <Image2, Z2, C2, ,..., ImageN, ZN, CN>);");
	}
	
	const int cameras = (in-_offset)/3;
	
	//if(cameras < 2)
	//{
	//	mexErrMsgTxt("At least two camera frames (and matrixes) must be provided!"); 
	//}

	if((in-_offset) % 3)
	{
		mexErrMsgTxt("Number of Images, Z-maps and Camera Matrixes should correspond to each other!"); 
	}	

	const float * const cd = (float*)mxGetData(input[0]);
	const glm::mat4x3 Cd = glm::mat4x3(cd[0],cd[1],cd[2],cd[3],cd[4],cd[5],cd[6],cd[7],cd[8],cd[9],cd[10],cd[11]);	
	const glm::mat4 Cdinv = glm::inverse(glm::mat4(Cd));

	const float minZ = static_cast<float>(mxGetScalar(input[1]));	
	const float maxZ = static_cast<float>(mxGetScalar(input[2]));	
	const int layers = static_cast<int>(mxGetScalar(input[3]));	
	const float thr = (maxZ-minZ)/(layers-1);
				
	const int height = mxGetDimensions(input[_offset])[0];		
	const int width = mxGetDimensions(input[_offset])[1];
	//const int colors = mxGetDimensions(input[_offset])[2];
	const long HW = width*height;

	//! validate the input data
	for(int n=0; n<cameras; n++)
	{
		if(mxGetClassID(input[_offset+n*3]) != mxSINGLE_CLASS)
		{
			mexPrintf("Camera Image %d must be of a SINGLE-type", n+1);
			mexErrMsgTxt("\n");			
		}
		
		if((mxGetDimensions(input[_offset+n*3]))[0] != height || (mxGetDimensions(input[_offset+n*3]))[1] != width || (mxGetDimensions(input[_offset+n*3]))[2] != colors)
		{			
			mexErrMsgTxt("All Camera Images must have the same resolution and three colors!");
		}

		if(mxGetClassID(input[_offset+n*3 + 1]) != mxSINGLE_CLASS)
		{
			mexPrintf("Camera Z-map %d must be of a SINGLE-type", n+1);
			mexErrMsgTxt("\n");			
		}
		
		if((mxGetDimensions(input[_offset+n*3 + 1]))[0] != height || (mxGetDimensions(input[_offset+n*3 + 1]))[1] != width)
		{		
			mexPrintf("Camera Z-map %d must be of the same resolution, but not %d x %d!", n+1, (mxGetDimensions(input[_offset+n*3 + 1]))[1] , (mxGetDimensions(input[_offset+n*3 + 1]))[0]);
			mexErrMsgTxt("\n");
		}

		if(mxGetClassID(input[_offset+n*3+2]) != mxSINGLE_CLASS)
		{
			mexPrintf("Camera Matrix %d must be of a SINGLE-type", n+1);
			mexErrMsgTxt("\n");
		}

		if((mxGetDimensions(input[_offset+n*3+2]))[0] != 3 || (mxGetDimensions(input[_offset+n*3+2]))[1] != 4)
		{
			mexPrintf("Camera matrix %d must be of size 3x4", n+1);
			mexErrMsgTxt("\n");
		}
	}

	//MexImage<float>** Images = new MexImage<float>*[cameras];
	std::unique_ptr<MexImage<float>*[]> Images(new MexImage<float>*[cameras]);
	std::unique_ptr<MexImage<float>*[]> Depths(new MexImage<float>*[cameras]);
	//glm::mat4x3 *C = new glm::mat4x3[cameras];
	//glm::mat4 *Cinv = new glm::mat4[cameras];

	//typedef std::unique_ptr<MexImage<float>> SafeImage;
	//std::unique_ptr<SafeImage[]> Images (new SafeImage[cameras]);
	std::unique_ptr<glm::mat4x3[]> C(new glm::mat4x3[cameras]);	
	//std::unique_ptr<float[]> distances(new float[cameras]);
	
	
	#pragma omp parallel for
	for(int n=0; n<cameras; n++)
	{		
		Images[n] = new MexImage<float>(input[_offset + n*3]);
		Depths[n] = new MexImage<float>(input[_offset + n*3 + 1]);
		const float * const c1 = (float*)mxGetData(input[_offset + n*3 + 2]);
		C[n] = glm::mat4x3(c1[0],c1[1],c1[2],c1[3],c1[4],c1[5],c1[6],c1[7],c1[8],c1[9],c1[10],c1[11]);	
	}
	
	const float nan = sqrt(-1.f);	
	const mwSize dims3[] = {(unsigned)height, (unsigned)width, colors};
	const mwSize dims[] = {(unsigned)height, (unsigned)width, 1};
	
	//Matlab-allocated variables	
	output[0] = mxCreateNumericArray(3, dims3, mxSINGLE_CLASS, mxREAL);
	output[1] = mxCreateNumericArray(3, dims, mxSINGLE_CLASS, mxREAL);
	MexImage<float> ImageV(output[0]);
	MexImage<float> DepthV(output[1]);

	
	MexImage<float> BestCost(width, height);
	BestCost.setval(cost_thr);

	// for each depth-layer in the space
	#pragma omp parallel 
	{
		//MexImage<float> ColorTmpl(width, height, colors);
		MexImage<float> Color(width, height, colors);
		//MexImage<float> CostTmp(width, height);
		MexImage<float> Cost(width, height);		
		MexImage<float> PrevCost(width, height);		
		PrevCost.setval(thr*4);
		//MexImage<float> Weight(width, height);
		

		#pragma omp for		
		//for (int d = 0; d < layers; d++)
		for (int d = layers-1; d >= 0; d--)
		{
			//float z = 1.f / ((float(d) / layers)*(1.f / minZ - 1.f / maxZ) + 1.f / maxZ);
			const float z = minZ + d*(maxZ-minZ)/(layers-1);
			MexImage<float> &ImageRef = *Images[0];
			//Cost.setval(0);
			//Color.setval(0);
			//Weight.setval(0);
			for (int n = 0; n < cameras; n ++)
			{
				Color.setval(0);
				Cost.setval(thr);				

				MexImage<float> &Image = *Images[n];			
				MexImage<float> &Depth = *Depths[n];
				projectImage(Color, Cdinv, Image, C[n], z);				
				projectDepth(Cost, Cdinv, Depth, C[n], z, thr*4);
				//Cost.IntegralImage(true);

				#pragma omp parallel for
				for (long i = 0; i < HW; i++)
				{
					const int x = i / height;
					const int y = i % height;

					//const float cost = Cost.getIntegralAverage(x,y,3);
					const float cost = Cost[i];
					//if(cost <= BestCost[i] )
					//if(cost <= 2*thr)
					if(cost <= PrevCost[i] && cost <= 2*thr) 
					{
						#pragma omp critical
						{
							BestCost[i] = cost;
							DepthV[i] = z;						
							ImageV[i] = Color[i];
							ImageV[i+HW] = Color[i+HW];
							ImageV[i+2*HW] = Color[i+2*HW];
						}
					} 

					PrevCost[i] = Cost[i];
				}

				
			}

					//const float w = cost_thr/(CostTmp[i] < 0.1 ? 0.1 : CostTmp[i]);
					//Cost[i] += CostTmp[i];
					//Weight[i] += w;
					//for(int c=0; c<colors; c++)
					//{
					//	Color[i + c*HW] += ColorTmpl[i + c*HW] * w;						
					//}
				

			//#pragma omp parallel for
			//for (long i = 0; i < HW; i++)
			//{
			//	if(Cost[i] <= BestCost[i])
			//	{
			//		#pragma omp critical
			//		{
			//			BestCost[i] = Cost[i];
			//			DepthV[i] = z;						
			//			ImageV[i] = ColorTmpl[i]/Weight[i];
			//			ImageV[i+HW] = ColorTmpl[i+HW]/Weight[i];
			//			ImageV[i+2*HW] = ColorTmpl[i+2*HW]/Weight[i];
			//		}
			//	}
			//}
		}
	}

	for(int n=0; n<cameras; n++)
	{		
		delete Images[n];
		delete Depths[n];
	}

}
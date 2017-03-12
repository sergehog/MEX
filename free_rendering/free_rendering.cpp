/** free_rendering
* @author Sergey Smirnov
* @date 30.01.2015
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


#define LIM(A,B) ((A) < 0 ? 0 : ((A) > (B)-1 ? (B)-1 : (A)))

void projectImage_CLAMP_TO_EDGE(MexImage<float> &Desired, const glm::mat4 Cdinv, MexImage<float> &Image, const glm::mat4x3 Ci, const float z)
{
	const int d_width = Desired.width;
	const int d_height = Desired.height;
	const int i_width = Image.width;
	const int i_height = Image.height;
	const int colors = Desired.layers;
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
				
			//#pragma loop(hint_parallel(colors))
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

#define cost_thr 5
void projectImageDepth(MexImage<float> &DesiredImage, MexImage<float> &Cost, const glm::mat4 Cdi, MexImage<float> &Image, MexImage<float> &Depth, const glm::mat4x3 C1, const float z)
{
	const int d_width = DesiredImage.width;
	const int d_height = DesiredImage.height;
	const int i_width = Image.width;
	const int i_height = Image.height;
	const int colors = DesiredImage.layers;

	const long dHW = d_width * d_height;

	for (int ud = 0; ud<d_width; ud++)
	{
		for (int vd = 0; vd<d_height; vd++)
		{
			long indexd = Image.Index(ud, vd);
			const glm::vec3 uvzi = C1 * (Cdi * glm::vec4(ud*z, vd*z, z, 1));
			//const glm::vec3 uvzi = Cd * (Cinv * glm::vec4(ud*z, vd*z, z, 1));
			const float zi = uvzi.z;
			const float ui = uvzi.x / uvzi.z;
			const float vi = uvzi.y / uvzi.z;
			const int ul = std::floor(ui);
			const int vl = std::floor(vi);
			
			if (ul<0 || ul >= i_width - 2 || vl < 0 || vl >= i_height - 2)
			{
				continue;
			}

			// nearest value
			const int ur = glm::round(ui);
			const int vr = glm::round(vi);
			//if (ur<0 || ur >= i_width - 1 || vr < 0 || vr >= i_height - 1)
			//{
			//	continue;
			//}

			const float dx = ui - ul;
			const float dy = vi - vl;

			for (int c = 0; c<colors; c++)
			{
				DesiredImage(ud, vd, c) = Image(ul, vl, c)*(1 - dx)*(1 - dy) + Image(ul + 1, vl, c)*dx*(1 - dy) + Image(ul, vl + 1, c)*(1 - dx)*dy + Image(ul + 1, vl + 1, c)*dx*dy;
			}
			
			//const float cost = abs(zi - Depth(ur, vr));
			//Cost(ud, vd) = cost > cost_thr ? cost_thr : cost;
			Cost(ud, vd) = abs(zi - Depth(ur, vr));
		}
	}
}

#define r 1

void mexFunction(int nout, mxArray* output[], int in, const mxArray* input[])
{
#ifndef _DEBUG
	omp_set_num_threads(std::max(4, omp_get_max_threads()));
	omp_set_dynamic(std::max(2, omp_get_max_threads() / 2));
#endif
	const int offset = 1;
	if (in < offset+3 /*|| !((in-offset)%3)*/ || nout != 3 || mxGetClassID(input[offset]) != mxSINGLE_CLASS || mxGetClassID(input[offset+1]) != mxSINGLE_CLASS || mxGetClassID(input[offset+2]) != mxSINGLE_CLASS)
	{
		mexErrMsgTxt("USAGE: [Img, Z, W] = free_rendering(C_desired, single(Img1), single(Z1), single(C1) <, ..., ImgN, ZN, CN>);");
	}

	const float sigma = static_cast<float>(mxGetScalar(input[1]));
	const float z_thr = static_cast<float>(mxGetScalar(input[2]));
	//const float maxZ = static_cast<float>(mxGetScalar(input[2]));
	//const int layers = static_cast<int>(mxGetScalar(input[3]));
	const float nan = std::sqrt(-1);
	
	const float * const cd = (float*)mxGetData(input[0]);	
	const glm::mat4x3 Cd(cd[0], cd[1], cd[2], cd[3], cd[4], cd[5], cd[6], cd[7], cd[8], cd[9], cd[10], cd[11]);
	//const glm::mat4 Cdi = glm::inverse(glm::mat4(Cd));
	

	const int cameras = (in - offset)/3;
	std::unique_ptr<glm::mat4x3[]> Matrices = std::unique_ptr<glm::mat4x3[]>(new glm::mat4x3[cameras]);
	std::unique_ptr<glm::mat4[]> Inverses = std::unique_ptr<glm::mat4[]>(new glm::mat4[cameras]);
	std::unique_ptr<MexImage<float>*[]> Images = std::unique_ptr<MexImage<float>*[]>(new MexImage<float>*[cameras]);
	std::unique_ptr<MexImage<float>*[]> Depths = std::unique_ptr<MexImage<float>*[]>(new MexImage<float>*[cameras]);

	for(int i=0; i<cameras; i++)
	{
		Images[i] = new MexImage<float>(input[offset + i*3]);
		Depths[i] = new MexImage<float>(input[offset + i*3 + 1]);
		const float * const c1 = (float*)mxGetData(input[offset + i*3 + 2]);	
		glm::mat4x3 Ci = glm::mat4x3(c1[0], c1[1], c1[2], c1[3], c1[4], c1[5], c1[6], c1[7], c1[8], c1[9], c1[10], c1[11]);
		Matrices[i] = Ci;
		Inverses[i] = glm::inverse(glm::mat4(Ci));
	}

	const int width = Images[0]->width;
	const int height = Images[0]->height;
	const int colors = Images[0]->layers;
	const long HW = width * height;
	
	const mwSize dimC[] = { (unsigned)height, (unsigned)width, (unsigned)colors };
	const mwSize dimZ[] = { (unsigned)height, (unsigned)width, 1};

	output[0] = mxCreateNumericArray(3, dimC, mxSINGLE_CLASS, mxREAL);
	output[1] = mxCreateNumericArray(3, dimZ, mxSINGLE_CLASS, mxREAL);
	output[2] = mxCreateNumericArray(3, dimZ, mxSINGLE_CLASS, mxREAL);
	
	MexImage<float> Rendered(output[0]);
	MexImage<float> ZBuffer(output[1]);
	MexImage<float> Weights(output[2]);
	MexImage<float> TextureUV(width, height, 2);

	Weights.setval(0);
	Rendered.setval(nan);
	ZBuffer.setval(nan);
	TextureUV.setval(nan);

	//#pragma omp parallel
	{
		
		//MexImage<float> Weights(width, height);
		//MexImage<float> ProjectedDepth(width, height);
		
		//std::unique_ptr<MexImage<float>*[]> Projecteds = std::unique_ptr<MexImage<float>*[]>(new MexImage<float>*[cameras]);
		//std::unique_ptr<MexImage<float>*[]> Costs = std::unique_ptr<MexImage<float>*[]>(new MexImage<float>*[cameras]);

		//for(int i=0; i<cameras; i++)
		//{
		//	Projecteds[i] = new MexImage<float>(width, height);
		//	Costs[i] = new MexImage<float>(width, height);
		//}

		//#pragma omp for
		
		for(int camera=0; camera<cameras; camera++)
		{
			//MexImage<float> &Projected = *Projecteds[camera];
			//MexImage<float> &Cost = *Costs[camera];
			MexImage<float> &Image =*Images[camera];
			MexImage<float> &Depth =*Depths[camera];
			//ProjectedDepth.setval(nan);
			//Weights.setval(0);
			glm::mat4 Ci_inv = glm::inverse(glm::mat4(Matrices[camera]));
			#pragma omp parallel for
			for(long i=0; i<HW; i++)
			{
				const int ui = i / height;
				const int vi = i % height;
				const float zi = Depth[i];
			
				if(isnan(zi))
				{
					continue;
				}
				const glm::vec3 uvzd = Cd * (Ci_inv * glm::vec4(ui*zi, vi*zi, zi, 1));
				const float zd = uvzd.z;
				const float ud = uvzd.x / zd;
				const float vd = uvzd.y / zd;
				const int ul = std::floor(ud);
				const int vl = std::floor(vd);

				for(int u=ul-r; u<=ul+r; u++)
				{
					for(int v=vl-r; v<=vl+r; v++)
					{
						if(u<0 || u>= width || v<0 || v>=height)
						{
							continue;
						}
						//float w = exp(-sqrt((u-ud)*(u-ud)+(v-vd)*(v-vd))/2);
						//float w = 1 - sqrt((u-ud)*(u-ud)+(v-vd)*(v-vd));
						float w = 1.41 - sqrt((u-ud)*(u-ud)+(v-vd)*(v-vd));
						w = w < 0 ? 0 : w;
						if(isnan(ZBuffer(u,v)) || zd < 0.98 * ZBuffer(u,v) && w > 0.1)//Weights(u,v))
						{
							ZBuffer(u,v) = zd;
							Weights(u,v) = w;
							TextureUV(u,v,0) = ui*w;
							TextureUV(u,v,1) = vi*w;
						}
						else if(zd >= 0.98 * ZBuffer(u,v) && zd <= 1.02 * ZBuffer(u,v))
						{
							if(w > Weights(u,v))
							{
								ZBuffer(u,v) = zd;
							}
							Weights(u,v) += w;
							TextureUV(u,v,0) += ui*w;
							TextureUV(u,v,1) += vi*w;
						}						
					}
				}
			}

			#pragma omp parallel for
			for(long i=0; i<HW; i++)
			{
				const int x = i / height;
				const int y = i % height;

				if(!isnan(ZBuffer[i]) && Weights[i] > 0.1f)
				{
					float u = TextureUV(x,y,0)/Weights(x,y);
					float v = TextureUV(x,y,1)/Weights(x,y);
					const int ul = std::floor(u);
					const int vl = std::floor(v);
			
					if (ul<0 || ul >= width - 2 || vl < 0 || vl >= height - 2)
					{
						continue;
					}

					const float dx = u - ul;
					const float dy = v - vl;
					
					for (int c = 0; c<colors; c++)
					{
						Rendered(x, y, c) = Image(ul, vl, c)*(1 - dx)*(1 - dy) + Image(ul + 1, vl, c)*dx*(1 - dy) + Image(ul, vl + 1, c)*(1 - dx)*dy + Image(ul + 1, vl + 1, c)*dx*dy;
					}
				}
				else
				{
					ZBuffer[i] = nan;
				}
			}
		}
		
	}
	

	for(int i=0; i<cameras; i++)
	{
		delete Images[i];
		delete Depths[i];
	}

	
}
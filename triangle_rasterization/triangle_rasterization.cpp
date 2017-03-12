/**
*	@file triangle_rasterization.cpp
*	@date 06.06.2016
*	@author Sergey Smirnov <sergey.smirnov@tut.fi>
*/

#ifdef __cplusplus 
extern "C" {
#include "mex.h"
}
#endif // __cplusplus
#define GLM_FORCE_CXX11  
#include <glm/glm.hpp>
#include <cmath>
#include <algorithm>
#include <memory>
#include "../common/meximage.h"


typedef unsigned char uint8;
using namespace mymex;

#define isnan _isnan

#pragma comment (lib, "libeng.lib") 
#pragma comment (lib, "libmx.lib")
#pragma comment (lib, "libmex.lib")

inline float min3(const float a1, const float a2, const float a3)
{
	return  a1 < a2 ? (a1 < a3 ? a1 : a3) : (a2 < a3 ? a2 : a3);
}

inline float max3(const float a1, const float a2, const float a3)
{
	return  a1 > a2 ? (a1 > a3 ? a1 : a3) : (a2 > a3 ? a2 : a3);
}

float orient2d(const float ax, const float ay, const float bx, const float by, const int x, const int y)
{
	return (bx - ax)*(y - ay) - (by - ay)*(x - ax);
}

float orient2d(glm::vec3 a, glm::vec3 b, const int x, const int y)
{
	return (b.x - a.x)*(y - a.y) - (b.y - a.y)*(x - a.x);
}


void mexFunction(int nout, mxArray* output[], int in, const mxArray* input[])
{
	if (in != 6 || nout != 2 || mxGetClassID(input[0]) != mxUINT64_CLASS || mxGetClassID(input[1]) != mxSINGLE_CLASS)
	{
		mexErrMsgTxt("USAGE: [Image, Depth] = triangle_rasterization(uint64(Faces), single(Vertices), width, height, UV_attribs_or_Colors, Texture); ");
	}

	MexImage<uint64_t> Faces(input[0]);
	MexImage<float> Vertices(input[1]);
	if (Faces.width != 3 || Faces.height < 4)
	{
		mexErrMsgTxt("Faces must be a UINT64 matrix of Nx3 size!");
	}
	if (Vertices.width != 3 || Vertices.height < 4)
	{
		mexErrMsgTxt("Vertices must be a SINGLE matrix of Kx3 size!");
	}

	const int faces = Faces.height;
	const int vertices = Vertices.height;
	uint64_t minF = 100;
	#pragma omp parallel for
	for (int f=0; f<faces; f++)
	{		
		if (Faces(0, f) < minF)
		{
			#pragma omp atomic
			minF = Faces(0, f);
		}
		if (Faces(1, f) < minF)
		{
			#pragma omp atomic
			minF = Faces(1, f);
		}
		if (Faces(2, f) < minF)
		{
			#pragma omp atomic
			minF = Faces(2, f);
		}
	}

	const int width = (int)mxGetScalar(input[2]);
	const int height = (int)mxGetScalar(input[3]);
	
	MexImage<float> UV(input[4]);
	MexImage<float> Texture(input[5]);	
	const int colors = Texture.layers;
	const int texture_width = Texture.width;
	const int texture_height = Texture.height;
	if (UV.height != Vertices.height && !(UV.width == 2 || UV.width == 3))
	{
		mexErrMsgTxt("UV_attribs must have same height as Vertices!");
	}

	size_t dims3[] = { (unsigned)height, (unsigned)width, colors};
	size_t dims[] = { (unsigned)height, (unsigned)width, 1};

	output[0] = mxCreateNumericArray(3, dims3, mxSINGLE_CLASS, mxREAL);
	output[1] = mxCreateNumericArray(3, dims, mxSINGLE_CLASS, mxREAL);
	
	MexImage<float> Image(output[0]);
	MexImage<float> Depth(output[1]);
	Image.setval(NAN);
	Depth.setval(NAN);


	#pragma omp parallel for
	for (int face = 0; face < faces; face++)
	{
		const uint64_t _v0 = Faces(0, face) - minF;
		const uint64_t _v1 = Faces(1, face) - minF;
		const uint64_t _v2 = Faces(2, face) - minF;
		if (_v0 >= vertices || _v1 >= vertices || _v1 >= vertices)
		{
			continue;
		}

		const glm::vec3 v0 = glm::vec3(Vertices(0, _v0), Vertices(1, _v0), Vertices(2, _v0));
		const glm::vec3 v1 = glm::vec3(Vertices(0, _v1), Vertices(1, _v1), Vertices(2, _v1));
		const glm::vec3 v2 = glm::vec3(Vertices(0, _v2), Vertices(1, _v2), Vertices(2, _v2));

		const int minX = std::max<int>(0, int(floor(min3(v0.x, v1.x, v2.x))));
		const int minY = std::max<int>(0, int(floor(min3(v0.y, v1.y, v2.y))));
		const int maxX = std::min<int>(width - 1, int(ceil(max3(v0.x, v1.x, v2.x))));
		const int maxY = std::min<int>(height - 1, int(ceil(max3(v0.y, v1.y, v2.y))));

		//const int minX = std::max<int>(0, int(floor(_minX)));
		//const int minY = std::max<int>(0, int(floor(_minY)));
		//const int maxX = std::min<int>(width-1, int(ceil(_maxX)));
		//const int maxY = std::min<int>(height-1, int(ceil(_maxY)));

		for (int x = minX; x <= maxX; x++)
		{
			for (int y = minY; y <= maxY; y++)
			{				
				//const float w0 = orient2d(v1, v2, x, y);
				//const float w1 = orient2d(v2, v0, x, y);
				//const float w2 = orient2d(v0, v1, x, y);

				const float _w0 = orient2d(Vertices(0, _v1), Vertices(1, _v1), Vertices(0, _v2), Vertices(1, _v2), x, y);
				const float _w1 = orient2d(Vertices(0, _v2), Vertices(1, _v2), Vertices(0, _v0), Vertices(1, _v0), x, y);
				const float _w2 = orient2d(Vertices(0, _v0), Vertices(1, _v0), Vertices(0, _v1), Vertices(1, _v1), x, y);
				// face culling is disabled here 
				const float w0 = _w0 >= 0 ? _w0 : -_w0;
				const float w1 = _w0 >= 0 ? _w1 : -_w1;
				const float w2 = _w0 >= 0 ? _w2 : -_w2;

				if ( w0 >= 0.f && w1 >= 0.f && w2 >= 0.f  && !isnan(w0) && !isnan(w1) && !isnan(w2))
				{
					const float w = abs(w0) + abs(w1) + abs(w2);
					const float newDepth = abs(w)<1e-5 ? v0.z : 1.f / ((w0 / v0.z + w1 / v1.z + w2 / v2.z) / w);
					const glm::vec2 st0 = glm::vec2(UV(0, _v0), UV(1, _v0));
					const glm::vec2 st1 = glm::vec2(UV(0, _v1), UV(1, _v1));
					const glm::vec2 st2 = glm::vec2(UV(0, _v2), UV(1, _v2));

					const glm::vec2 st = abs(w)<1e-5 ? st0 : (st0*w0 + st1*w1 + st2*w2) / w;
					//const float s = (w == 0.f) ? s0 : (s0*w0 + s1*w1 + s2*w2) / w;
					//const float t = (w == 0.f) ? t0 : (t0*w0 + t1*w1 + t2*w2) / w;
					//const float ray_norm = abs(x) + abs(y) + 
					//const glm::vec3 ray();
					

					if (newDepth > 0.f && (isnan(Depth(x, y)) || newDepth < Depth(x, y)))
					{
						Depth(x, y) = newDepth;
						
						if (UV.width == 2)
						{
							for (int c = 0; c < colors; c++)
							{
								const int sf = std::min<int>(Texture.width - 1, std::max<int>(0, int(floor(st.x))));
								const int tf = std::min<int>(Texture.height - 1, std::max<int>(0, int(floor(st.y))));
								const int sc = std::min<int>(Texture.width - 1, std::max<int>(0, int(ceil(st.x))));
								const int tc = std::min<int>(Texture.height - 1, std::max<int>(0, int(ceil(st.y))));
								const float sw = st.x - sf;
								const float tw = st.y - tf;
								const float color = Texture(sf, tf, c)*(1 - sw)*(1 - tw) + Texture(sc, tf, c)*(sw)*(1 - tw) + Texture(sc, tc, c)*(sw)*(tw)+Texture(sf, tc, c)*(1 - sw)*(tw);
								Image(x, y, c) = color;
							}
						}
						else if (UV.width == 3)
						{
							for (int c = 0; c < 3; c++)
							{
								const float color0 = UV(c, _v0);
								const float color1 = UV(c, _v1);
								const float color2 = UV(c, _v2);
								Image(x, y, c) = (color0*w0 + color1*w1 + color2*w2)/(w0+w1+w2);
							}
						}

					}
				}
			}
		}

	}

}


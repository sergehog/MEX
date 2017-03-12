// triangle_rasterization_pong.cpp 
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
		mexErrMsgTxt("USAGE: [Image, Depth] = triangle_rasterization_pong(uint64(Faces), single(Vertices), width, height, UV_attribs, Texture); ");
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
	
	const float phong = 1.f;
	const glm::vec3 light_source(0, 0, 0);
	const float light_color[4] = { 255.f, 255.f, 255.f, 1.f };


	const int faces = Faces.height;
	const int vertices = Vertices.height;
	uint64_t minF = 100;
#pragma omp parallel for
	for (int f = 0; f<faces; f++)
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
	if (UV.height != Vertices.height || UV.width != 2)
	{
		mexErrMsgTxt("UV_attribs must have same height as Vertices!");
	}

	size_t dims3[] = { (unsigned)height, (unsigned)width, colors };
	size_t dims[] = { (unsigned)height, (unsigned)width, 1 };

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

		// calculate normal 
		const glm::vec3 v = v1 - v0;
		const glm::vec3 w = v2 - v0;
		const float n0 = v.y*w.z - v.z*w.y;
		const float n1 = v.x*w.x - v.x*w.z;
		const float n2 = v.x*w.y - v.y*w.x;
		const glm::vec3 normal = glm::normalize(glm::vec3(n0, n1, n2));
		
		const glm::vec3 light = glm::normalize(light_source - glm::vec3(v0.x, v0.y, v0.z));
		const float cosine = std::max(0.f, glm::dot(normal, light));


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
				const float w0 = _w0 > 0 ? _w0 : -_w0;
				const float w1 = _w0 > 0 ? _w1 : -_w1;
				const float w2 = _w0 > 0 ? _w2 : -_w2;

				if (w0 >= 0.f && w1 >= 0.f && w2 >= 0.f && !isnan(w0) && !isnan(w1) && !isnan(w2))
				{
					const float w = w0 + w1 + w2;
					const float newDepth = (w == 0.f) ? v0.z : 1.f / ((w0 / v0.z + w1 / v1.z + w2 / v2.z) / w);
					const glm::vec2 st0 = glm::vec2(UV(0, _v0), UV(1, _v0));
					const glm::vec2 st1 = glm::vec2(UV(0, _v1), UV(1, _v1));
					const glm::vec2 st2 = glm::vec2(UV(0, _v2), UV(1, _v2));

					const glm::vec2 st = (w == 0.f) ? st0 : (st0*w0 + st1*w1 + st2*w2) / w;
					//const float s = (w == 0.f) ? s0 : (s0*w0 + s1*w1 + s2*w2) / w;
					//const float t = (w == 0.f) ? t0 : (t0*w0 + t1*w1 + t2*w2) / w;

					if (newDepth > 0.f && (isnan(Depth(x, y)) || newDepth < Depth(x, y)))
					{
						Depth(x, y) = newDepth;
						//const float cosine = glm::dot(normal, light);
						for (int c = 0; c < colors; c++)
						{
							const int sf = int(floor(st.x));
							const int tf = int(floor(st.y));
							const int sc = int(ceil(st.x));
							const int tc = int(ceil(st.y));
							const float sw = st.x - sf;
							const float tw = st.y - tf;
							//const float color = Texture(sf, tf, c)*(1 - sw)*(1 - tw) + Texture(sc, tf, c)*(sw)*(1 - tw) + Texture(sc, tc, c)*(sw)*(tw)+Texture(sf, tc, c)*(1 - sw)*(tw);							
							Image(x, y, c) = light_color[c] * cosine;
						}
					}
				}
			}
		}

	}

}


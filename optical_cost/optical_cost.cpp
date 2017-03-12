/** optical_cost
* @author Sergey Smirnov
* @date 10.04.2015
*/
#ifdef __cplusplus 
extern "C" {
#include "mex.h"
}
#endif // __cplusplus
#include "../common/meximage.h"
#include <algorithm>
#include <array>
#include <memory>
#ifndef _NDEBUG
#include <omp.h>
#endif

#pragma comment (lib, "libeng.lib") 
#pragma comment (lib, "libmx.lib")
#pragma comment (lib, "libmex.lib")
#pragma warning(disable:4244)
#pragma warning(disable:4018)

using namespace mymex;
#define MAXRADIUS 20
#define MAXCOLORS 10

// base class needed for polimorfic matcher call 

struct base_matcher
{
	virtual void compute_cost(MexImage<float> &Ref, MexImage<float> &Tmpl, MexImage<float> &Cost) {};
};

template<unsigned radius, unsigned colors>
struct cost_matcher : base_matcher
{
	static const unsigned diameter = radius * 2 + 1;
	static const unsigned window = diameter*diameter;

	void compute_cost(MexImage<float> &Ref, MexImage<float> &Tmpl, MexImage<float> &Cost)
	{
		const int height = Ref.height;
		const int width = Ref.width;
		const long HW = width*height;
		const float nan = sqrt(-1);
		
		//Cost.setval(FLT_MAX);

		#pragma omp parallel for
		for (int w = 0; w<int(window); w++)
		{
			const int dy = w % int(diameter) - (int)radius;
			const int dx = w / int(diameter) - (int)radius;

			#pragma omp parallel for    
			for (long i = 0; i<HW; i++)
			{
				const int y = i % height;
				const int x = i / height;
				
				const int x2 = (x + dx) < 0 ? 0 : (x + dx) >= width ? width - 1 : (x + dx);
				const int y2 = (y + dy) < 0 ? 0 : (y + dy) >= height ? height - 1 : (y + dy);
				
				float difference = 0;
				
				for (int c = 0; c<colors; c++)
				{
					difference += abs(float(Ref(x, y, c)) - float(Tmpl(x2, y2, c)));
				}

				Cost(x, y, w) = difference;
			}
		}
	}
};

//template <typename T>
//struct matcher_array
//{
//	typedef std::array<base_matcher<T>*, (MAXCOLORS)*(MAXRADIUS)> type;
//};

typedef std::array<base_matcher*, (MAXCOLORS)*(MAXRADIUS)> matcher_array;

template<unsigned radius, unsigned colors>
class unroll_radius
{	
	unroll_radius<radius - 1, colors> unroller;
	cost_matcher<radius, colors> *matcher;

public:
	unroll_radius(matcher_array & matchers) : unroller(matchers)
	{
		matcher = new cost_matcher<radius, colors>();
		matchers[(colors-1)*MAXRADIUS + (radius - 1)] = (base_matcher*)matcher;
	}

	~unroll_radius()
	{
		delete matcher;
	}
};

template<unsigned colors>
class unroll_radius<0, colors>
{	
public:
	unroll_radius(matcher_array & matchers)
	{
	}
};

template<unsigned radius, unsigned colors>
class unroll_color
{
	unroll_radius<radius, colors> radius_unroller;
	unroll_color<radius, colors - 1> color_unroller;

public:
	unroll_color(matcher_array & matchers) : radius_unroller(matchers), color_unroller(matchers)
	{
	}
};

template<unsigned radius>
class unroll_color<radius,0>
{
public:
	unroll_color(matcher_array & matchers)
	{
	}
};



void mexFunction(int nout, mxArray* output[], int in, const mxArray* input[])
{
#ifndef _NDEBUG
	omp_set_num_threads(std::max<int>(4, omp_get_max_threads()));
	omp_set_dynamic(std::max<int>(2, omp_get_max_threads() / 2));
#endif

	if (in != 3 || nout != 1)
	{
		mexErrMsgTxt("USAGE: [CostRefxy] = optical_cost(Reference, Template, radius);");
	}

	const int radius = static_cast<int>(mxGetScalar(input[2]));	

	const mxClassID type0 = mxGetClassID(input[0]);
	const mxClassID type1 = mxGetClassID(input[1]);
	const int width = mxGetWidth(input[0]);
	const int height = mxGetHeight(input[0]);
	const int colors = mxGetLayers(input[0]);

	if (height != mxGetHeight(input[1]) || width != mxGetWidth(input[1]) || colors != mxGetLayers(input[1]))
	{
		mexErrMsgTxt("ERROR: Sizes of parameters 'left', 'right' must be the same.");
	}

	if (colors < 1 || colors > 9)
	{
		mexErrMsgTxt("Too many colors in your images.");
	}
	
	const size_t diameter = radius * 2 + 1;
	const size_t window = diameter*diameter;
	const mwSize costsize[] = { (size_t)height, (size_t)width, window};

	const bool bothFloat = (type0 == mxSINGLE_CLASS && type1 == mxSINGLE_CLASS);
	const bool bothByte = (type0 == mxUINT8_CLASS && type1 == mxUINT8_CLASS);
	if (!bothFloat && !bothByte)
	{
		mexErrMsgTxt("Both input images must be either floats or uint8.");
	}

	output[0] = mxCreateNumericArray(3, costsize, mxSINGLE_CLASS, mxREAL);

	MexImage<float> CostRef(output[0]);
	
	if (radius == 0 ||radius > MAXRADIUS || colors == 0 || colors>MAXCOLORS)
	{
		mexErrMsgTxt("Search radius or number of colors is too high.");
		return;
	}

	if (bothFloat)
	{
		matcher_array matchers;		
		unroll_color<MAXRADIUS, MAXCOLORS> unroller(matchers);

		MexImage<float> Ref(input[0]);
		MexImage<float> Templ(input[1]);		
		matchers[(colors-1)*MAXRADIUS + (radius-1)]->compute_cost(Ref, Templ, CostRef);
		
	}	
	//else if (bothByte)
	//{
	//	matcher_array<unsigned char>::type matchers;
	//	unroll_color<unsigned char, MAXRADIUS, MAXCOLORS> unroller(matchers);

	//	MexImage<unsigned char> Ref(input[0]);
	//	MexImage<unsigned char> Templ(input[1]);
	//	matchers[(colors - 1)*MAXRADIUS + (radius - 1)]->compute_cost(Ref, Templ, CostRef);		
	//}
	else
	{
		mexErrMsgTxt("Something goes wrong.");
	}

}
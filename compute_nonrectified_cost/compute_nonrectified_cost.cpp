/** compute_nonrectified_cost
* @author Sergey Smirnov
* @date 1.03.2013
*/

#ifdef __cplusplus 
extern "C" {
#include "mex.h"
}
#endif // __cplusplus

#include "../common/meximage.h"
#include "../common/mymath.h"
#include <cmath>
#include <algorithm>
#include <omp.h>
#include <limits>

typedef unsigned char uint8;

#pragma comment (lib, "libeng.lib") 
#pragma comment (lib, "libmx.lib")
#pragma comment (lib, "libmex.lib")
//#pragma warning(disable:4244)
//#pragma warning(disable:4018)

using namespace mymex;
using namespace mymath;

void mexFunction (int nout, mxArray* output[], int in, const mxArray* input[])
{
	omp_set_num_threads(std::max(4,omp_get_num_threads())); 
	omp_set_dynamic(0);

	if(in < 7 || in > 8 || nout != 2 || mxGetClassID(input[0]) != mxSINGLE_CLASS || mxGetClassID(input[1]) != mxSINGLE_CLASS || mxGetClassID(input[2]) != mxSINGLE_CLASS || mxGetClassID(input[3]) != mxSINGLE_CLASS)
	{
		mexErrMsgTxt("USAGE: [CostL, <CostR>] = compute_nonrectified_cost(single(Left), single(Right), P_left, P_right, mindepth, maxdepth, layers, [cost_threshold]);"); 
    } 

	if(!MexImage<float>::isValidType(input[0]) || !MexImage<float>::isValidType(input[1]))
	{
		mexErrMsgTxt("ERROR: Only SINGLE type is allowed for Left and Right."); 
	}

	MexImage<float> Left(input[0]);
	MexImage<float> Right(input[1]);
	Matrix<4, 4> P_left(input[2]);
	Matrix<4, 4> P_right(input[3]);

	Matrix<4, 4> P_left_invert;
	Matrix<4, 4> P_right_invert;

	invertMatrix4x4(&P_left, &P_left_invert);
	invertMatrix4x4(&P_left, &P_right_invert);

	const int width = Left.width;
	const int height = Left.height;
	const int colors = Left.layers;
	const int HW = Left.layer_size;

	if(width != Right.width || height != Right.height || colors != Right.layers)
	{
		mexErrMsgTxt("ERROR: Sizes of Left and  Right must be the same!"); 
	}	

	const float mindepth = static_cast<float>(mxGetScalar(input[4]));	
	const float maxdepth = static_cast<float>(mxGetScalar(input[5]));	
	const int layers = std::max(static_cast<int>(mxGetScalar(input[6])),2);
	//const float nan = (float)std::numeric_limits<float>::quiet_NaN ;
	const float nan = sqrt(-1.f);
	const float cost_threshold = (in > 7) ?  static_cast<float>(mxGetScalar(input[7])) : 255.f;	
	const size_t depthcost[] = {(unsigned)height, (unsigned)width, (unsigned)layers};

	//Matlab-allocated variables
	output[0] = mxCreateNumericArray(3, depthcost, mxSINGLE_CLASS, mxREAL);
	output[1] = mxCreateNumericArray(3, depthcost, mxSINGLE_CLASS, mxREAL);

	MexImage<float> CostL(output[0]);
	CostL.setval(nan);
    MexImage<float> CostR(output[1]);
	CostR.setval(nan);

	#pragma omp parallel
	{
		Matrix<4, 1> pointLeft(0.f);
		float *color = new float[colors];

		#pragma omp for
		for(long i=0; i<HW; i++)
		{
			int x = i / height;
			int y = i % height;

			for(int layer = 0; layer < layers; layer ++)
			{
				for(int c=0; c<colors; c++)
					color[c] = 0.f;				

				float z = 1.0/((layer/(float)layers)*(1.0/mindepth - 1.0/maxdepth) + 1.0/maxdepth);				

				float z = maxdepth - layer*(maxdepth-mindepth)/layers;
				float yv = height - y - 1;
				pointLeft.data[0][0] = float(x)*z;
				pointLeft.data[1][0] = float(yv)*z;
				pointLeft.data[2][0] = z;
				pointLeft.data[3][0] = 1.f;

				Matrix<4, 1> pointWorld =  P_left_invert.multiply<1>(pointLeft);
				Matrix<4, 1> pointRightVirtual = P_right.multiply<1>(pointWorld);

				float zv = pointRightVirtual.data[2][0];
				if(zv <= 0.f)
					continue;
				
				
				float xv = pointRightVirtual.data[0][0]/zv;
				float yv = pointRightVirtual.data[1][0]/zv;
				yv = height - yv - 1;

				if(!(0.f <= xv && xv < (float) width && 0.f <= yv && yv < (float) height))
				{
					continue;
				}

				int xvi = floor(xv);
				int yvi = floor(yv);
				float dl = xv-(float)xvi;
				float du = yv-(float)yvi;

				long ilu = CostL.Index(xvi, yvi);
				long iru = CostL.Index(xvi+1, yvi);
				long ild = CostL.Index(xvi, yvi+1);
				long ird = CostL.Index(xvi+1, yvi+1);						

				float weights = 0;
				float weight = sqrt(dl*dl + du*du);						
				for(int c=0; c<colors; c++)
				{
					color[c] += Left[ilu + HW*c]*weight;
				}
				weights += weight;

				weight = sqrt((1-dl)*(1-dl) + du*du);						
				for(int c=0; c<colors; c++)
				{
					color[c] += Left[iru + HW*c]*weight;
				}
				weights += weight;

				weight = sqrt(dl*dl + (1-du)*(1-du));						
				for(int c=0; c<colors; c++)
				{
					color[c] += Left[ild + HW*c]*weight;
				}
				weights += weight;

				weight = sqrt((1-dl)*(1-dl) + (1-du)*(1-du));						
				for(int c=0; c<colors; c++)
				{
					color[c] += Left[ird + HW*c]*weight;
				}
				weights += weight;

				for(int c=0; c<colors; c++)
				{
					color[c] /= weights;
				}


				float cost = 0;

				for(int c=0; c<colors; c++)
				{
					cost += abs(color[c] - Left[i + c*HW]);					 
				}	
				cost /= colors;

				CostL.data[i + layer*HW] = cost > cost_threshold ? cost_threshold : cost;

			}
			
		}
	}


}

 
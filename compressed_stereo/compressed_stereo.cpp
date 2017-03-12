/** Approximate depth with continous surface of low dimention
* @file compressed_stereo
* @date 23.06.2016
* @author Sergey Smirnov
*/
#ifdef __cplusplus 
extern "C" {
#include "mex.h"
}
#endif // __cplusplus
//#include "../common/common.h"
#include "../common/meximage.h"
#include <cstdint>
#include <algorithm>
#include <memory>
#include <omp.h>


#pragma comment (lib, "libeng.lib") 
#pragma comment (lib, "libmx.lib")
#pragma comment (lib, "libmex.lib")
#pragma warning(disable:4244)
#pragma warning(disable:4018)

constexpr int colors = 3;
constexpr int hypothesizes = 10;
//static const float nan = sqrt(-1.f);

using namespace mymex;

void upscalePyramid(MexImage<float>** Buffer, const int levels)
{
	//Buffer[levels]->copyFrom(Disp);
	
	// upscale disparity to the finest scale
	for (int l = levels - 1; l >= 0; l--)
	{
		MexImage<float> &DispCoarse = *Buffer[l + 1];
		MexImage<float> &DispFine = *Buffer[l];
		MexImage<float> DispMiddle(DispFine.width, DispCoarse.height);
		#pragma omp parallel for
		for (int y = 0; y < DispCoarse.height; y++)
		{
			for (int x = 0; x < DispFine.width; x++)
			{
				const int xc = x / 2;
				if (x % 2 && xc < DispCoarse.width - 1)
				{
					DispMiddle(x, y) = (DispCoarse(xc, y) + DispCoarse(xc + 1, y)) / 2.f;
				}
				else
				{
					DispMiddle(x, y) = DispCoarse(xc, y);
				}
			}
		}
		#pragma omp parallel for
		for (int x = 0; x < DispFine.width; x++)
		{
			for (int y = 0; y < DispFine.height; y++)
			{
				const int yc = y / 2;
				if (y % 2 && yc < DispCoarse.height - 1)
				{
					DispFine(x, y) = (DispMiddle(x, yc) + DispMiddle(x, yc + 1)) / 2.f;
				}
				else
				{
					DispFine(x, y) = DispMiddle(x, yc);
				}
			}
		}
	}
}

void downscalePyramid(MexImage<float>** Buffer, const int levels)
{
	for (int l = 1; l <= levels; l++)
	{
		MexImage<float> &ErrorCoarse = *Buffer[l];
		MexImage<float> &ErrorFine = *Buffer[l - 1];
		MexImage<float> ErrorMiddle(ErrorFine.width, ErrorCoarse.height);

		#pragma omp parallel for
		for (int x = 0; x < ErrorFine.width; x++)
		{
			for (int y = 0; y < ErrorCoarse.height; y++)
			{
				if (y * 2 < ErrorFine.height - 1)
				{
					ErrorMiddle(x, y) = (ErrorFine(x, y * 2) + ErrorFine(x, y * 2 + 1)) / 2;
				}
				else
				{
					ErrorMiddle(x, y) = ErrorFine(x, y * 2);
				}
			}
		}
		#pragma omp parallel for
		for (int y = 0; y < ErrorCoarse.height; y++)
		{
			for (int x = 0; x < ErrorCoarse.width; x++)
			{
				if (x * 2 < ErrorFine.width - 1)
				{
					ErrorCoarse(x, y) = (ErrorMiddle(x * 2, y) + ErrorMiddle(x * 2 + 1, y)) / 2;
				}
				else
				{
					ErrorCoarse(x, y) = ErrorMiddle(x * 2, y);
				}
			}
		}
	}
}

template<int dir>
void calculateCompressedError(const MexImage<const float> &Reference, const MexImage<const float> &Template, MexImage<float> &DispF, MexImage<float> &DispB, MexImage<float> &Error, MexImage<float> **Buffer, const int levels)
{
	const int width = Reference.width;
	const int height = Reference.height;
	MexImage<float> DispF_finest(width, height);
	MexImage<float> DispB_finest(width, height);
	Buffer[levels]->copyFrom(DispF);
	upscalePyramid(Buffer, levels);
	DispF_finest.copyFrom(*Buffer[0]);

	Buffer[levels]->copyFrom(DispB);
	upscalePyramid(Buffer, levels);
	DispB_finest.copyFrom(*Buffer[0]);

	calculateError<dir>(Reference, Template, DispF_finest, DispB_finest, *Buffer[0]);
	
	downscalePyramid(Buffer, levels);
	Error.copyFrom(*Buffer[levels]);
}

template<int dir>
void calculateError(const MexImage<const float> &Reference, const MexImage<const float> &Template, MexImage<float> &DispF, MexImage<float> &DispB, MexImage<float> &Error)
{	
	const int width = Reference.width;
	const int height = Reference.height;
	
	#pragma omp parallel for
	for(long i=0; i<Reference.layer_size; i++)
	{
		const int x = i / height;
		const int y = i % height;
		const float df = DispF(x, y);
		const float db = DispB(x, y);
		const float xf = std::min(width-1.f, std::max(0.f, x + dir*df));
		const int xf_f = static_cast<int>(floor(xf));
		const int xf_c = static_cast<int>(ceil(xf));
		const float wf = xf_c - xf;
		const float xb = std::min(width - 1.f, std::max(0.f, x + dir*db));
		const int xb_f = static_cast<int>(floor(xb));
		const int xb_c = static_cast<int>(ceil(xb));
		const float wb = xb_c - xb;

		float diffF = 0.f;
		float diffB = 0.f;
		for (int c = 0; c < colors; c++)
		{
			const float colorRf = Template(xf_f, y, c)*wf + Template(xf_c, y, c)*(1 - wf);
			diffF += abs(Reference(x, y, c) - colorRf);
			const float colorRb = Template(xb_f, y, c)*wb + Template(xb_c, y, c)*(1 - wb);
			diffB += abs(Reference(x, y, c) - colorRb);
		}
		Error(x, y) = diffF < diffB ? diffF : diffB;
	}
}

template<int dir>
void calculateError(const MexImage<const float> &Reference, const MexImage<const float> &Template, MexImage<float> &Disp, MexImage<float> &Error)
{
	const int width = Reference.width;
	const int height = Reference.height;

	#pragma omp parallel for
	for (long i = 0; i<Reference.layer_size; i++)
	{
		const int x = i / height;
		const int y = i % height;
		const float d = Disp(x, y);
		
		const float xr = std::min(width - 1.f, std::max(0.f, x + dir*d));
		const int xr_f = static_cast<int>(floor(xr));
		const int xr_c = static_cast<int>(ceil(xr));
		const float wr = xr_c - xr;

		float diff = 0.f;		
		for (int c = 0; c < colors; c++)
		{
			const float colorR = Template(xr_f, y, c)*wr + Template(xr_c, y, c)*(1 - wr);
			diff += abs(Reference(x, y, c) - colorR);
		}
		Error(x, y) = diff;
	}
}

void mexFunction(int nout, mxArray* output[], int in, const mxArray* input[])
{
#ifndef _NDEBUG
	omp_set_num_threads(std::max(8, omp_get_max_threads()));
	omp_set_dynamic(std::max(7, omp_get_max_threads() - 1));
	//omp_set_dynamic(omp_get_max_threads());
#endif	

	if (in < 4 || in > 6 || nout != 4 || mxGetClassID(input[0]) != mxSINGLE_CLASS || mxGetClassID(input[1]) != mxSINGLE_CLASS || mxGetClassID(input[2]) != mxSINGLE_CLASS || mxGetClassID(input[3]) != mxSINGLE_CLASS || mxGetClassID(input[4]) != mxLOGICAL_CLASS)
	{
		mexErrMsgTxt("USAGE: [DispFg2, DispBg2, ErrFg, ErrBg] = compressed_stereo(single(Reference), single(Template), levels, maxdisp, [mindisp = 0, iterations=2]);");
	}

	//const int direction = int(mxGetScalar(input[5])) == -1 ? -1 : 1;
	const int levels = std::max(1, static_cast<int>(mxGetScalar(input[2])));
	const int maxdisp = static_cast<int>(mxGetScalar(input[3]));
	const int mindisp = (in > 4) ? static_cast<int>(mxGetScalar(input[4])) : 0;
	const int iterations = (in > 5) ? std::max(1, static_cast<int>(mxGetScalar(input[5]))) : 2;
	
	if (maxdisp <= mindisp)
	{
		mexErrMsgTxt("ERROR: 'maxdisp' must be larger than 'mindisp'!");
	}

	const MexImage<const float> Left(input[0]);
	const MexImage<const float> Right(input[1]);
	const int width = Left.width;
	const int height = Left.height;
	const int64_t HW = static_cast<int64_t>(Left.layer_size);
	//const MexImage<const float> DispFg(input[2]);
	//const MexImage<const float> DispBg(input[3]);
	//const MexImage<const bool> Valid(input[4]);

	if (height != Right.height || width != Right.width || colors != Right.layers || colors != Left.layers)
	{
		mexErrMsgTxt("ERROR: Sizes of parameters 'Left', 'Right' must be the same.");
	}	

	if (colors < 1 || colors > 9)
	{
		mexErrMsgTxt("Too many colors in your images.");
	}

	const mwSize dims[] = { (size_t)height, (size_t)width, 1 };
	//const mwSize dimsCoarse[] = { (size_t)height_coarsest, (size_t)width_coarsest, 1 };
	
	output[0] = mxCreateNumericArray(3, dims, mxSINGLE_CLASS, mxREAL);
	output[1] = mxCreateNumericArray(3, dims, mxSINGLE_CLASS, mxREAL);	
	output[2] = mxCreateNumericArray(3, dims, mxSINGLE_CLASS, mxREAL);
	output[3] = mxCreateNumericArray(3, dims, mxSINGLE_CLASS, mxREAL);

	MexImage<float> DispLf_finest(output[0]);
	MexImage<float> DispLb_finest(output[1]);
	MexImage<float> ErrorF(output[2]);
	MexImage<float> ErrorB(output[3]);

	DispLf_finest.setval(-1.f);
	DispLb_finest.setval(-2.f);
	ErrorF.setval(FLT_MAX);
	ErrorB.setval(FLT_MAX);
		
	std::unique_ptr<MexImage<float>*[]> Buffer(new MexImage<float>*[levels + 1]);
	Buffer[0] = new MexImage<float>(width, height);

	for (int l = 1; l <= levels; l++)
	{
		const int width_finer = Buffer[l - 1]->width;
		const int height_finer = Buffer[l - 1]->height;
		const int width_coarse = width_finer % 2 ? width_finer / 2 + 1 : width_finer / 2;
		const int height_coarse = height_finer % 2 ? height_finer / 2 + 1 : height_finer / 2;
		Buffer[l] = new MexImage<float>(width_coarse, height_coarse);
	}

	const int width_coarsest = Buffer[levels]->width;
	const int height_coarsest = Buffer[levels]->height;
	const int HW_coarsest = width_coarsest * height_coarsest;
	
	MexImage<float> BestErrorL(width_coarsest, height_coarsest);	
	MexImage<float> BestDispF(width_coarsest, height_coarsest);
	MexImage<float> BestDispB(width_coarsest, height_coarsest);
	MexImage<float> DispLf(width_coarsest, height_coarsest);
	MexImage<float> DispLb(width_coarsest, height_coarsest);
	MexImage<float> ErrorL(width_coarsest, height_coarsest);
	//BestDispL.setval(0.f);
	BestErrorL.setval(FLT_MAX);
	BestDispF.setval(FLT_MAX);
	BestDispB.setval(FLT_MAX);
	
	//Buffer[0]->copyFrom(DispFg);
	//downscalePyramid(Buffer.get(), levels);
	//BestDispF.copyFrom(*Buffer[levels]);

	//Buffer[0]->copyFrom(DispBg);
	//downscalePyramid(Buffer.get(), levels);
	//BestDispB.copyFrom(*Buffer[levels]);



	//// Initialization 
	//for (int df = mindisp; df <= maxdisp; df++)
	//{
	//	DispLf.setval(df);
	//	
	//	for (int db = mindisp; db <= df; db++)
	//	{
	//		DispLb.setval(db);
	//		calculateCompressedError<-1>(Left, Right, DispLf, DispLb, ErrorL, Buffer.get(), levels);
	//		#pragma omp parallel for
	//		for (int i = 0; i < HW_coarsest; i++)
	//		{
	//			if (ErrorL[i] < BestErrorL[i])
	//			{
	//				BestErrorL[i] = ErrorL[i];
	//				BestDispLf[i] = df;
	//				BestDispLb[i] = db;
	//			}
	//		}
	//	}		
	//}

	
	MexImage<float> DispLfMAX(width, height);
	MexImage<float> DispLfMIN(width, height);
	MexImage<float> DispLbMAX(width, height);
	MexImage<float> DispLbMIN(width, height);

	for (int iter = 1; iter < iterations; iter++)
	{
		DispLf.copyFrom(BestDispLf);
		DispLb.copyFrom(BestDispLb);

		for (int i = 0; i < HW_coarsest; i++)
		{
			const int x = i / height_coarsest;
			const int y = i % height_coarsest;
			const float d0 = BestDispLf(x, y);
			const float dL = BestDispLf(std::max(0, x - 1), y);
			const float dR = BestDispLf(std::min(width-1, x + 1), y);
			const float dU = BestDispLf(x, std::max(0, y - 1));
			const float dD = BestDispLf(x, std::min(height - 1, y + 1));

		}

		for (int hi = 0; hi < hypothesizes; hi++)
		{


		}
	}
	//	for (int d = mindisp; d <= maxdisp; d++)
	//	{
	//		for (int i = 0; i < HW_coarsest; i++)
	//		{
	//			const int xc = i / height_coarsest;
	//			const int yc = i % height_coarsest;
	//			if (xc % 2 || yc % 2)
	//			{
	//				continue;
	//			}
	//			DispLCoarse(xc, yc) = d;
	//		}

	//		calculateError(Left, Right, DispLn.get(), ErrorsL.get(), levels);

	//		#pragma omp parallel for
	//		for (int i = 0; i < HW_coarsest; i++)
	//		{
	//			if (ErrorLCoarse[i] < BestErrorL[i])
	//			{
	//				BestErrorL[i] = ErrorLCoarse[i];
	//				BestDispL[i] = DispLCoarse[i];
	//			}
	//		}
	//	}

	//	DispLCoarse.copyFrom(BestDispL);

	//	for (int d = mindisp; d <= maxdisp; d++)
	//	{
	//		#pragma omp parallel for
	//		for (int i = 0; i < HW_coarsest; i++)
	//		{
	//			const int xc = i / height_coarsest;
	//			const int yc = i % height_coarsest;
	//			if (xc % 2 || !(yc % 2))
	//			{
	//				continue;
	//			}
	//			DispLCoarse(xc, yc) = d;
	//		}

	//		calculateError(Left, Right, DispLn.get(), ErrorsL.get(), levels);
	//		#pragma omp parallel for
	//		for (int i = 0; i < HW_coarsest; i++)
	//		{
	//			if (ErrorLCoarse[i] < BestErrorL[i])
	//			{
	//				BestErrorL[i] = ErrorLCoarse[i];
	//				BestDispL[i] = DispLCoarse[i];
	//			}
	//			//DispLCoarse[i] = BestDispL[i];
	//		}
	//	}

	//	DispLCoarse.copyFrom(BestDispL);

	//	for (int d = mindisp; d <= maxdisp; d++)
	//	{
	//		#pragma omp parallel for
	//		for (int i = 0; i < HW_coarsest; i++)
	//		{
	//			const int xc = i / height_coarsest;
	//			const int yc = i % height_coarsest;
	//			if (!(xc % 2) || (yc % 2))
	//			{
	//				continue;
	//			}
	//			DispLCoarse(xc, yc) = d;
	//		}

	//		calculateError(Left, Right, DispLn.get(), ErrorsL.get(), levels);

	//		#pragma omp parallel for
	//		for (int i = 0; i < HW_coarsest; i++)
	//		{
	//			if (ErrorLCoarse[i] < BestErrorL[i])
	//			{
	//				BestErrorL[i] = ErrorLCoarse[i];
	//				BestDispL[i] = DispLCoarse[i];
	//			}
	//			//DispLCoarse[i] = BestDispL[i];
	//		}
	//	}

	//	DispLCoarse.copyFrom(BestDispL);

	//	for (int d = mindisp; d <= maxdisp; d++)
	//	{
	//		#pragma omp parallel for
	//		for (int i = 0; i < HW_coarsest; i++)
	//		{
	//			const int xc = i / height_coarsest;
	//			const int yc = i % height_coarsest;
	//			if (!(xc % 2) || !(yc % 2))
	//			{
	//				continue;
	//			}
	//			DispLCoarse(xc, yc) = d;
	//		}

	//		calculateError(Left, Right, DispLn.get(), ErrorsL.get(), levels);
	//		#pragma omp parallel for
	//		for (int i = 0; i < HW_coarsest; i++)
	//		{
	//			if (ErrorLCoarse[i] < BestErrorL[i])
	//			{
	//				BestErrorL[i] = ErrorLCoarse[i];
	//				BestDispL[i] = DispLCoarse[i];
	//			}
	//			//DispLCoarse[i] = BestDispL[i];
	//		}
	//	}

	//	DispLCoarse.copyFrom(BestDispL);
	//}
	

	Buffer[levels]->copyFrom(BestDispLf);
	upscalePyramid(Buffer.get(), levels);
	DispLf_finest.copyFrom(*Buffer[0]);

	Buffer[levels]->copyFrom(BestDispLb);
	upscalePyramid(Buffer.get(), levels);
	DispLb_finest.copyFrom(*Buffer[0]);
	
	calculateError<-1>(Left, Right, DispLf_finest, *Buffer[0]);
	MexImage<float> Buffer2(width, height);
	calculateError<-1>(Left, Right, DispLb_finest, Buffer2);
	#pragma omp parallel for
	for (long i = 0; i < HW; i++)
	{
		if (Buffer2[i] < Buffer[0]->data[i])
		{
			DispLf_finest[i] = DispLb_finest[i];
		}
	}


	for (int l = 0; l <= levels; l++)
	{
		delete Buffer[l];
	}
	




}
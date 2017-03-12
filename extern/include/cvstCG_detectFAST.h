/* Copyright 2012 The MathWorks, Inc. */

#ifndef _DETECTFAST_
#define _DETECTFAST_

#ifndef LIBMWDETECTFAST_API
#    define LIBMWDETECTFAST_API
#endif

#ifndef EXTERN_C
#  ifdef __cplusplus
#    define EXTERN_C extern "C"
#  else
#    define EXTERN_C extern
#  endif
#endif

#include "tmwtypes.h"

EXTERN_C LIBMWDETECTFAST_API int32_T detectFAST_compute(uint8_T *inImg, 
	int32_T nRows, int32_T nCols, int32_T isRGB, 
	int threshold,
	void **outKeypoints);

EXTERN_C LIBMWDETECTFAST_API void detectFAST_assignOutput(void *ptrKeypoints,
	real32_T *outLoc, real32_T *outMetric);

#endif

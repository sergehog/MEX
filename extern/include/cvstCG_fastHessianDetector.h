/* Copyright 2012 The MathWorks, Inc. */

#ifndef _FASTHESSIANDETECTOR_
#define _FASTHESSIANDETECTOR_

#ifndef LIBMWFASTHESSIANDETECTOR_API
#    define LIBMWFASTHESSIANDETECTOR_API
#endif

#ifndef EXTERN_C
#  ifdef __cplusplus
#    define EXTERN_C extern "C"
#  else
#    define EXTERN_C extern
#  endif
#endif

#include "tmwtypes.h"

EXTERN_C LIBMWFASTHESSIANDETECTOR_API int32_T fastHessianDetector_uint8(uint8_T *inImg, 
	int32_T nRows, int32_T nCols, int32_T nDims, 
	int32_T nOctaveLayers, int32_T nOctaves, int32_T hessianThreshold,
	void **dptrKeypoints);

EXTERN_C LIBMWFASTHESSIANDETECTOR_API void fastHessianDetector_keyPoints2field(void *keypointsV,
	real32_T *outLoc, real32_T *outScale, real32_T *outMetric, int8_T *outSignOfLap);

EXTERN_C LIBMWFASTHESSIANDETECTOR_API void fastHessianDetector_deleteKeypoint(void *ptrKeypoints);
#endif

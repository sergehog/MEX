/* Copyright 2012 The MathWorks, Inc. */

#ifndef _DISPARITYBM_
#define _DISPARITYBM_

#ifndef LIBMWDISPARITYBM_API
#    define LIBMWDISPARITYBM_API
#endif

#ifndef EXTERN_C
#  ifdef __cplusplus
#    define EXTERN_C extern "C"
#  else
#    define EXTERN_C extern
#  endif
#endif

#include "tmwtypes.h"

#ifndef typedef_cvstDBMStruct_T
#define typedef_cvstDBMStruct_T

 typedef struct {
	int preFilterCap; 
	int SADWindowSize; 
	int minDisparity; 
	int numberOfDisparities; 
	int textureThreshold; 
	int uniquenessRatio; 
	int disp12MaxDiff; 
	const int32_T roi1[4]; 
	const int32_T roi2[4]; 
	int preFilterType; 
	int preFilterSize; 
	int speckleWindowSize; 
	int speckleRange;
	int trySmallerWindows;
} cvstDBMStruct_T;

#endif /*typedef_cvstDBMStruct_T: used by matlab coder*/

EXTERN_C LIBMWDISPARITYBM_API void disparityBM_compute(
	const uint8_T* inImg1, const uint8_T* inImg2, int nRows, int nCols, 
	real32_T* dis,
	cvstDBMStruct_T *params);

#endif

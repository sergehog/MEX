/* Copyright 2012 The MathWorks, Inc. */

#ifndef _DETECTBRISK_
#define _DETECTBRISK_

#ifndef LIBMWDETECTBRISK_API
#    define LIBMWDETECTBRISK_API
#endif

#ifndef EXTERN_C
#  ifdef __cplusplus
#    define EXTERN_C extern "C"
#  else
#    define EXTERN_C extern
#  endif
#endif

#include "tmwtypes.h"

EXTERN_C LIBMWDETECTBRISK_API
int32_T detectBRISK_detect(uint8_T *img, 
                           int nRows, int nCols,
                           int threshold, int numOctaves,
                           void **outKeypoints);

EXTERN_C LIBMWDETECTBRISK_API
void detectBRISK_assignOutputs(void *ptrKeypoints,
                               real32_T * location,real32_T * metric,
                               real32_T * scale, real32_T * orientation);

#endif

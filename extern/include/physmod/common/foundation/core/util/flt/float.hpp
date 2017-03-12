/*********************************************************************************************************************
 ** File: float.hpp
 ** Abstract:
 **     Implement utility functions for manipulating floating point numbers.
 **
 ** Copyright 2005-2008 The MathWorks, Inc.
 *********************************************************************************************************************/

#ifndef __float_hpp__
#define __float_hpp__


/*!
 * return utGetInf()
 */
EXTERN_C double pmf_get_inf(void);


/*!
 * return utDoubleScalarIsfinite(x)
 */
EXTERN_C bool pmf_is_finite(double x);


/*!
 * return utIsInf(x)
 */
EXTERN_C bool pmf_is_inf(double x);


/*!
 * return utIsNan(x)
 */
EXTERN_C bool pmf_is_nan(double x);


/*!
 * return utGetNan()
 */
EXTERN_C double pmf_get_nan(void);


/*!
 * return utGetEps(x)
 */
EXTERN_C double pmf_get_eps(void);


/*!
 * return utGetRealMax(void)
 */
EXTERN_C double pmf_get_real_max(void);


/*!
 * return utGetRealMin(void)
 */
EXTERN_C double pmf_get_real_min(void);


/*!
 * return utGetPI(void)
 */
EXTERN_C double pmf_get_pi(void);


#endif /* include guard */

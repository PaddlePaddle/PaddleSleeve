#ifndef _MATH_H_
#define _MATH_H_

/******************************************************************************
 * Floating point data types                                                  *
 ******************************************************************************/

/*  Define float_t and double_t per C standard, ISO/IEC 9899:2011 7.12 2,
    taking advantage of GCC's __FLT_EVAL_METHOD__ (which a compiler may
    define anytime and GCC does) that shadows FLT_EVAL_METHOD (which a
    compiler must define only in float.h).                                    */
#if __FLT_EVAL_METHOD__ == 0
    typedef float float_t;
    typedef double double_t;
#elif __FLT_EVAL_METHOD__ == 1
    typedef double float_t;
    typedef double double_t;
#elif __FLT_EVAL_METHOD__ == 2 || __FLT_EVAL_METHOD__ == -1
    typedef long double float_t;
    typedef long double double_t;
#else /* __FLT_EVAL_METHOD__ */
#   error "Unsupported value of __FLT_EVAL_METHOD__."
#endif /* __FLT_EVAL_METHOD__ */

extern float expf (float x);
extern float fabsf(float x);
extern float roundf(float x);
extern double round(double x);

namespace std {

template <typename T>
inline T const& max(T const& a, T const& b) {
  return a < b ? b : a;
}

template <typename T>
inline T const& min(T const& a, T const& b) {
  return a < b ? a : b;
}

} //  namespace std

#endif /* _MATH_H_ */

#ifndef _LIBM_H_
#define _LIBM_H_

#include <math.h>
#include <sys/types.h>

typedef unsigned int uint32_t;
#ifdef _FLT_LARGEST_EXPONENT_IS_NORMAL
#define FLT_UWORD_IS_NAN(x) 0
#define FLT_UWORD_IS_INFINITE(x) 0
#define FLT_UWORD_LOG_MAX 0x42b2d4fc
#else
#define FLT_UWORD_IS_NAN(x) ((x)>0x7f800000L)
#define FLT_UWORD_IS_INFINITE(x) ((x)==0x7f800000L)
#define FLT_UWORD_LOG_MAX 0x42b17217
#endif

#ifdef _FLT_NO_DENORMALS
#define FLT_UWORD_LOG_MIN 0x42aeac50
#else
#define FLT_UWORD_LOG_MIN 0x42cff1b5
#endif

typedef union
{
  float value;
  uint32_t word;
} ieee_float_shape_type;

/* Get a 32 bit int from a float.  */

#define GET_FLOAT_WORD(i,d)                 \
do {                                \
  ieee_float_shape_type gf_u;                   \
  gf_u.value = (d);                     \
  (i) = gf_u.word;                      \
} while (0)

/* Set a float from a 32 bit int.  */

#define SET_FLOAT_WORD(d,i)                 \
do {                                \
  ieee_float_shape_type sf_u;                   \
  sf_u.word = (i);                      \
  (d) = sf_u.value;                     \
} while (0)

#define __math_oflowf(x) (x ? -0x1p97f : 0x1p97f) * 0x1p97f
#define __math_uflowf(x) (x ? -0x1p-95f : 0x1p-95f) * 0x1p-95f

#endif /* _LIBM_H_ */

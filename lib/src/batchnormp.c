#include <TH/TH.h>
#include "batchnormp.h"

#define THNN_CHECK_SHAPE(I1, I2)			\
  if (I1 != NULL && I2 != NULL && !THFloatTensor_isSameSizeAs(I1, I2))	\
    {							\
       THDescBuff s1 = THFloatTensor_sizeDesc(I1);		\
       THDescBuff s2 = THFloatTensor_sizeDesc(I2);		\
       THError(#I1 " and " #I2 " shapes do not match: "	\
	       #I1 " %s, " #I2 " %s", s1.str, s2.str);	\
    }

void BatchNormalizationP_forward(
  THFloatTensor *input, THFloatTensor *output,
  THFloatTensor *weight, THFloatTensor *bias,
  THFloatTensor *running_mean, THFloatTensor *running_var,
  THFloatTensor *save_mean, THFloatTensor *save_std,
  int train, double momentum, double eps)
{
  THFloatTensor_resizeAs(output, input);
  int64_t nInput = THFloatTensor_size(input, 1);
  int64_t f;
  ptrdiff_t n = THFloatTensor_nElement(input) / nInput;

  #pragma omp parallel for
  for (f = 0; f < nInput; ++f) {
    THFloatTensor *in = THFloatTensor_newSelect(input, 1, f);
    THFloatTensor *out = THFloatTensor_newSelect(output, 1, f);

    float mean, invstd, std;

    if (train) {
      // compute mean per input
//      double sum = 0;
//      TH_TENSOR_APPLY(float, in, sum += *in_data;);
//
//      mean = (float) sum / n;
//      THFloatTensor_set1d(save_mean, f, (float) mean);

      mean = THFloatTensor_get1d(save_mean, f);
      std = THFloatTensor_get1d(save_std, f);
      invstd = (float) (1 / (std + eps));

      // compute variance per input
//      sum = 0;
//      TH_TENSOR_APPLY(float, in,
//        sum += (*in_data - mean) * (*in_data - mean););
//
//      if (sum == 0 && eps == 0.0) {
//        invstd = 0;
//      } else {
//        invstd = (float) (1 / sqrt(sum/n + eps));
//      }
//      THFloatTensor_set1d(save_std, f, (float) invstd);

      // update running averages
      THFloatTensor_set1d(running_mean, f,
        (float) (momentum * mean + (1 - momentum) * THFloatTensor_get1d(running_mean, f)));

      double unbiased_var = std * n / (n - 1);
      THFloatTensor_set1d(running_var, f,
        (float) (momentum * unbiased_var + (1 - momentum) * THFloatTensor_get1d(running_var, f)));
    } else {
      mean = THFloatTensor_get1d(running_mean, f);
      invstd = 1 / sqrt(THFloatTensor_get1d(running_var, f) + eps);
    }

    // compute output
    float w = weight ? THFloatTensor_get1d(weight, f) : 1;
    float b = bias ? THFloatTensor_get1d(bias, f) : 0;

    TH_TENSOR_APPLY2(float, in, float, out,
      *out_data = (float) (((*in_data - mean) * invstd) * w + b););

    THFloatTensor_free(out);
    THFloatTensor_free(in);
  }
}

void BatchNormalizationP_backward(
  THFloatTensor *input, THFloatTensor *gradOutput, THFloatTensor *gradInput,
  THFloatTensor *gradWeight, THFloatTensor *gradBias, THFloatTensor *weight,
  THFloatTensor *running_mean, THFloatTensor *running_var,
  THFloatTensor *save_mean, THFloatTensor *save_std,
  int train, double scale, double eps)
{
  THNN_CHECK_SHAPE(input, gradOutput);
  int64_t nInput = THFloatTensor_size(input, 1);
  int64_t f;
  ptrdiff_t n = THFloatTensor_nElement(input) / nInput;

  #pragma omp parallel for
  for (f = 0; f < nInput; ++f) {
    THFloatTensor *in = THFloatTensor_newSelect(input, 1, f);
    THFloatTensor *gradOut = THFloatTensor_newSelect(gradOutput, 1, f);
    float w = weight ? THFloatTensor_get1d(weight, f) : 1;
    float mean, invstd;
    if (train) {
      mean = THFloatTensor_get1d(save_mean, f);
      invstd = 1 / (THFloatTensor_get1d(save_std, f) + eps);
    } else {
      mean = THFloatTensor_get1d(running_mean, f);
      invstd = 1 / sqrt(THFloatTensor_get1d(running_var, f) + eps);
    }

    // sum over all gradOutput in feature plane
    double sum = 0;
    TH_TENSOR_APPLY(float, gradOut, sum += *gradOut_data;);

    // dot product of the Q(X) and gradOuput
    double dotp = 0;
    TH_TENSOR_APPLY2(float, in, float, gradOut,
      dotp += (*in_data - mean) * (*gradOut_data););

    if (gradInput) {
      THFloatTensor_resizeAs(gradInput, input);
      THFloatTensor *gradIn = THFloatTensor_newSelect(gradInput, 1, f);

      if (train) {
        // when in training mode
        // Q(X) = X - E[x] ; i.e. input centered to zero mean
        // Y = Q(X) / σ    ; i.e. BN output before weight and bias
        // dL/dX = (Q(dL/dY) - dot(Y, dL/dY) * Y) / σ * w

        // projection of gradOutput on to output scaled by std
        float k = (float) dotp * invstd * invstd / n;
        TH_TENSOR_APPLY2(float, gradIn, float, in,
          *gradIn_data = (*in_data - mean) * k;);

        double gradMean = sum / n;
        TH_TENSOR_APPLY2(float, gradIn, float, gradOut,
          *gradIn_data = (*gradOut_data - gradMean - *gradIn_data) * invstd * w;);

      } else {
        // when in evaluation mode
        // Q(X) = X - running_mean  ; i.e. input centered to zero mean
        // Y = Q(X) / running_std    ; i.e. BN output before weight and bias
        // dL/dX = w / running_std
        TH_TENSOR_APPLY2(float, gradIn, float, gradOut,
          *gradIn_data = *gradOut_data * invstd * w;);
      }

      THFloatTensor_free(gradIn);
    }

    if (gradWeight) {
      float val = THFloatTensor_get1d(gradWeight, f);
      THFloatTensor_set1d(gradWeight, f, val + scale * dotp * invstd);
    }

    if (gradBias) {
      float val = THFloatTensor_get1d(gradBias, f);
      THFloatTensor_set1d(gradBias, f, val + scale * sum);
    }

    THFloatTensor_free(gradOut);
    THFloatTensor_free(in);
  }
}
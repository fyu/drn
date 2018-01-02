// #include "auto_gpu.h"
#include <THC/THC.h>

  #include "batchnormp_cuda_kernel.h"


extern THCState *state;

void BatchNormalizationP_forward_cuda(
  THCudaTensor *input, THCudaTensor *output,
  THCudaTensor *weight, THCudaTensor *bias,
  THCudaTensor *running_mean, THCudaTensor *running_var,
  THCudaTensor *save_mean, THCudaTensor *save_std,
  int train, double momentum, double eps) {
  THNN_CudaBatchNormalization_updateOutputhaha(
    state, input, output, weight, bias, running_mean, running_var,
    save_mean, save_std, train, momentum, eps);
}


void BatchNormalizationP_backward_cuda(
  THCudaTensor *input, THCudaTensor *gradOutput,
  THCudaTensor *gradOutputMean, THCudaTensor *dotP,
  THCudaTensor *gradInput,
  THCudaTensor *gradWeight, THCudaTensor *gradBias, THCudaTensor *weight,
  THCudaTensor *running_mean, THCudaTensor *running_var,
  THCudaTensor *save_mean, THCudaTensor *save_std,
  int train, double scale, double eps) {
  THNN_CudaBatchNormalization_backwardhaha(
      state, input, gradOutput, gradOutputMean, dotP,
      gradInput, gradWeight, gradBias, weight,
      running_mean, running_var, save_mean, save_std, train, scale, eps);
}
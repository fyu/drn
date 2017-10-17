#include <THC/THCTensor.h>

void THNN_CudaBatchNormalization_updateOutputhaha(
  THCState *state, THCudaTensor *input_, THCudaTensor *output_,
  THCudaTensor *weight_, THCudaTensor *bias_, THCudaTensor *runningMean_,
  THCudaTensor *runningVar_, THCudaTensor *saveMean_, THCudaTensor *saveStd_,
  int train, double momentum, double eps);


void THNN_CudaBatchNormalization_backwardhaha(
  THCState *state, THCudaTensor *input_, THCudaTensor *gradOutput_,
  THCudaTensor *gradOutputMean_, THCudaTensor *dotP,
  THCudaTensor *gradInput_, THCudaTensor *gradWeight_, THCudaTensor *gradBias_,
  THCudaTensor *weight_, THCudaTensor *runningMean_, THCudaTensor *runningVar_,
  THCudaTensor *saveMean_, THCudaTensor *saveStd_, int train, double scale,
  double eps);
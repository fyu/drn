void BatchNormalizationP_forward_cuda(
  THCudaTensor *input, THCudaTensor *output,
  THCudaTensor *weight, THCudaTensor *bias,
  THCudaTensor *running_mean, THCudaTensor *running_var,
  THCudaTensor *save_mean, THCudaTensor *save_std,
  int train, double momentum, double eps);


void BatchNormalizationP_mean_cuda(
  THCudaTensor *input, THCudaTensor *save_mean);


void BatchNormalizationP_var_cuda(
  THCudaTensor *input, THCudaTensor *save_mean, THCudaTensor *save_var);


void BatchNormalizationP_backward_cuda(
  THCudaTensor *input, THCudaTensor *gradOutput,
  THCudaTensor *gradOutputMean, THCudaTensor *dotP,
  THCudaTensor *gradInput,
  THCudaTensor *gradWeight, THCudaTensor *gradBias, THCudaTensor *weight,
  THCudaTensor *running_mean, THCudaTensor *running_var,
  THCudaTensor *save_mean, THCudaTensor *save_std,
  int train, double scale, double eps);


void BatchNormalizationP_mean_grad_cuda(
  THCudaTensor *input, THCudaTensor *gradOutput,
  THCudaTensor *runningMean, THCudaTensor *saveMean,
  THCudaTensor *gradOutputMean, THCudaTensor *dotP, int train);
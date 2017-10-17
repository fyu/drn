// #include <TH/TH.h>

void BatchNormalizationP_forward(
  THFloatTensor *input, THFloatTensor *output,
  THFloatTensor *weight, THFloatTensor *bias,
  THFloatTensor *running_mean, THFloatTensor *running_var,
  THFloatTensor *save_mean, THFloatTensor *save_std,
  int train, double momentum, double eps);


void BatchNormalizationP_backward(
  THFloatTensor *input, THFloatTensor *gradOutput, THFloatTensor *gradInput,
  THFloatTensor *gradWeight, THFloatTensor *gradBias, THFloatTensor *weight,
  THFloatTensor *running_mean, THFloatTensor *running_var,
  THFloatTensor *save_mean, THFloatTensor *save_std,
  int train, double scale, double eps);

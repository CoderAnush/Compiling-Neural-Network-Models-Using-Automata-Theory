# Example Report — Neural Network Compiler Demo

This is an example `report.md` showing the expected content and format produced by `python project/demo_output.py`.

## Graph (original)

{
  "Input": [
    "Conv_features_0"
  ],
  "Conv_features_0": [
    "ReLU_features_1"
  ],
  "ReLU_features_1": [
    "MaxPool_features_2"
  ],
  "MaxPool_features_2": [
    "Conv_features_3"
  ],
  "Conv_features_3": [
    "ReLU_features_4"
  ],
  "ReLU_features_4": [
    "MaxPool_features_5"
  ],
  "MaxPool_features_5": [
    "Flatten_features_6"
  ],
  "Flatten_features_6": [
    "Dense_classifier_0"
  ],
  "Dense_classifier_0": [
    "ReLU_classifier_1"
  ],
  "ReLU_classifier_1": [
    "Dense_classifier_2"
  ],
  "Dense_classifier_2": [
    "Softmax_classifier_3"
  ],
  "Softmax_classifier_3": [
    "Output"
  ],
  "Output": []
}

## IR (original)

%1 = Input()
%2 = Conv_features_0(%1)
%3 = ReLU_features_1(%2)
%4 = MaxPool_features_2(%3)
%5 = Conv_features_3(%4)
%6 = ReLU_features_4(%5)
%7 = MaxPool_features_5(%6)
%8 = Flatten_features_6(%7)
%9 = Dense_classifier_0(%8)
%10 = ReLU_classifier_1(%9)
%11 = Dense_classifier_2(%10)
%12 = Softmax_classifier_3(%11)
%13 = Output(%12)

## Graph (optimized)

{
  "Input": [
    "FusedConvReLU_features_0"
  ],
  "FusedConvReLU_features_0": [
    "MaxPool_features_2"
  ],
  "MaxPool_features_2": [
    "FusedConvReLU_features_3"
  ],
  "FusedConvReLU_features_3": [
    "MaxPool_features_5"
  ],
  "MaxPool_features_5": [
    "ReshapeDense_classifier_0"
  ],
  "ReshapeDense_classifier_0": [
    "ReLU_classifier_1"
  ],
  "ReLU_classifier_1": [
    "Dense_classifier_2"
  ],
  "Dense_classifier_2": [
    "Softmax_classifier_3"
  ],
  "Softmax_classifier_3": [
    "Output"
  ],
  "Output": []
}

## Benchmark (example)

{
  "layers_before": 13,
  "layers_after": 10,
  "speed_original": 0.69,
  "speed_optimized": 1.05,
  "speed_original_std": 0.01,
  "speed_optimized_std": 0.02,
  "boost": -52.74
}

---

Notes:
- This is a small example showing the format and types of values you can expect. Actual numbers will vary depending on hardware and whether weights are preserved.

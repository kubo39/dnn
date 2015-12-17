module dnn;

import std.stdio;
import util;


class NeuralNetwork
{
  double[] hiddenLayer;
  double[] inputLayer;
  double[] outputLayer;
  double[][] weightHidden;
  double[][] weightOutput;
  double[] errOutput;
  double[] errHidden;
  double[][] lastChangeHidden;
  double[][] lastChangeOutput;
  double rate1;
  double rate2;

  this(uint iInputCount,
       uint iHiddenCount,
       uint iOutputCount,
       double iRate1 = 0.25,
       double iRate2 = 0.1)  // Rateはランダムに与える重み
  in {
    assert(iInputCount >= 1, "can't have any empty layers.");
    assert(iHiddenCount >= 1, "can't have any empty layers.");
    assert(iOutputCount >= 1, "can't have any empty layers.");
  }
  body {
    rate1 = iRate1;
    rate2 = iRate2;

    inputLayer.length = iInputCount;
    hiddenLayer.length = iHiddenCount;
    outputLayer.length = iOutputCount;
    errOutput.length = iOutputCount;
    errHidden.length = iHiddenCount;

    weightHidden = randomMatrix(iHiddenCount, iInputCount, -1.0, 1.0);
    weightOutput = randomMatrix(iOutputCount, iHiddenCount, -1.0, 1.0);

    lastChangeHidden = makeMatrix(iHiddenCount, iInputCount, 0.0);
    lastChangeOutput = makeMatrix(iOutputCount, iHiddenCount, 0.0);
  }

  double[] forward(double[] input)
  {
    for (int i; i < input.length; ++i) {
      inputLayer[i] = input[i];
    }
    inputLayer[inputLayer.length-1] = 1.0;

    // 隠れ層の計算
    for (int i; i < hiddenLayer.length-1; ++i) {
      double sum = 0.0;
      for (int j; j < inputLayer.length; ++j) {
        sum += inputLayer[j] * weightHidden[i][j];
      }
      hiddenLayer[i] = sigmoid(sum);
    }
    hiddenLayer[hiddenLayer.length-1] = 1.0;

    // 出力層の計算
    for (int i; i < outputLayer.length; ++i) {
      double sum = 0.0;
      for (int j; j < hiddenLayer.length; ++j) {
        sum += hiddenLayer[j] * weightOutput[i][j];
      }
      outputLayer[i] = sigmoid(sum);
    }
    return outputLayer;
  }

  void feedback(double[] target)
  {
    for (int i; i < outputLayer.length; ++i) {
      errOutput[i] = outputLayer[i] - target[i];
    }

    for (int i; i < hiddenLayer.length-1; ++i) {
      double err = 0.0;
      for (int j; j < outputLayer.length; ++j) {
        err += errOutput[j] * weightOutput[j][i] * dsigmoid(outputLayer[j]);
      }
      errHidden[i] = err;
    }

    // 出力層の重み更新
    for (int i; i < outputLayer.length; ++i) {
      for (int j; j < hiddenLayer.length; ++j) {
        double change = 0.0;
        double delta = 0.0;
        delta = errOutput[i] * dsigmoid(outputLayer[i]);
        change = rate1 * delta * hiddenLayer[i] + rate2 * lastChangeOutput[i][j];
        weightOutput[i][j] -= change;
        lastChangeOutput[i][j] = change;
      }
    }

    // 隠れ層の重み計算
    for (int i; i < hiddenLayer.length-1; ++i) {
      for (int j; j < inputLayer.length; ++j) {
        auto delta = errHidden[i] * dsigmoid(hiddenLayer[i]);
        auto change = rate1 * delta * inputLayer[j] + rate2 * lastChangeHidden[i][j];
        weightHidden[i][j] -= change;
        lastChangeHidden[i][j] = change;
      }
    }
  }

  void train(double[][] inputs, double[][] targets, int iteration)
  {
    int iter_flag = -1;

    for (int i; i < iteration; ++i) {
      float cur_err = 0.0;
      for (int j; j < inputs.length; ++j) {
        auto idx_arr = genRandomIdx(inputs.length);
        forward(inputs[idx_arr[j]]);
        feedback(targets[idx_arr[j]]);

        // 二乗誤差の計算
        for (int k; k < outputLayer.length; ++k) {
          auto err = outputLayer[k] - targets[idx_arr[j]][k];
          cur_err += 0.5 * err * err;
        }

        debug {
          if ((j+1) % 1000 == 0) {
            if (iter_flag != i) {
              writeln("");
              iter_flag = i;
            }
            writefln("iteration %dth / progress %.2f", i+1, cast(double) j*100/cast(double) inputs.length);
          }
        }

      }

      debug {
        if ((iteration >= 10 && (i+1)%(iteration/10) == 0) || iteration < 10) {
          writefln("\niteration %dth MSE: %.5f", i+1, cur_err / cast(double) inputs.length);
        }
      }

    }
    debug writeln("done.");
  }
}

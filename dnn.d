import std.stdio, std.math, std.random;


double sigmoid(double x) {
  return 1.0 / (1.0 + pow(E, -x));
}


double dsigmoid(double y) {
  return y * (1.0 - y);
}


double[][] makeMatrix(int rows, int colums, double value) {
  double[][] mat;
  mat.length = rows;
  for (int i; i < rows; ++i) {
    mat[i].length = colums;
    for(int j; j < colums; ++j) {
      mat[i][j] = value;
    }
  }
  return mat;
}


double[][] randomMatrix(int rows, int colums, double lower, double upper) {
  double[][] mat;
  mat.length = rows;
  for (int i; i < rows; ++i) {
    mat[i].length = colums;
    for(int j; j < colums; ++j) {
      mat[i][j] = uniform(0.0, cast(double) (upper - lower)) + lower;
    }
  }
  return mat;
}


int[] genRandomIdx(int n) {
  int[] arr;
  arr.length = n;
  for (int i; i < n; ++i) {
    arr[i] = i;
  }

  for (int i; i < n; ++i) {
    auto j = i + cast(int) uniform(0.0, 1.0);
    auto tmp = arr[i];
    arr[i] = arr[j];
    arr[j] = tmp;
  }
  return arr;
}


class NeuralNetwork {

  double[] hiddenLayer;
  double[] inputLayer;
  double[] outputLayer;
  double[][] weightHidden;
  double[][] weightOutput;
  double[] errOutput;
  double[] errHidden;
  double[][] lastChangeHidden;
  double[][] lastChangeOutput;
  bool regression;
  double rate1;   // learning rate
  double rate2;

  this(int iInputCount, int iHiddenCount, int iOutputCount, bool iRegression,
       double iRate1 = 0.25, double iRate2 = 0.1) {
    iInputCount += 1;
    iHiddenCount += 1;

    regression = iRegression;
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

  double[] forward(double[] input) {
    for (int i; i < input.length; ++i) {
      inputLayer[i] = input[i];
    }
    inputLayer[inputLayer.length-1] = 1.0;

    for (int i; i < hiddenLayer.length-1; ++i) {
      double sum = 0.0;
      for (int j; j < inputLayer.length; ++j) {
	sum += inputLayer[j] * weightHidden[i][j];
      }
      hiddenLayer[i] = sigmoid(sum);
    }
    hiddenLayer[hiddenLayer.length-1] = 1.0;

    for (int i; i < outputLayer.length; ++i) {
      double sum = 0.0;
      for (int j; j < hiddenLayer.length; ++j) {
	sum += hiddenLayer[j] * weightOutput[i][j];
      }
      if (regression) {
	outputLayer[i] = sum;
      } else {
	outputLayer[i] = sigmoid(sum);
      }
    }
    return outputLayer;
  }

  void feedback(double[] target) {
    for (int i; i < outputLayer.length; ++i) {
      errOutput[i] = outputLayer[i] - target[i];
    }

    for (int i; i < hiddenLayer.length-1; ++i) {
      double err = 0.0;
      for (int j; j < outputLayer.length; ++j) {
	if (regression) {
	  err += errOutput[j] * weightOutput[j][i];
	} else {
	  err += errOutput[j] * weightOutput[j][i] * dsigmoid(outputLayer[j]);
	}
      }
      errHidden[i] = err;
    }

    for (int i; i < outputLayer.length; ++i) {
      for (int j; j < hiddenLayer.length; ++j) {
	double change = 0.0;
	double delta = 0.0;
	if (regression) {
	  delta = errOutput[i];
	} else {
	  delta = errOutput[i] * dsigmoid(outputLayer[i]);
	}
	change = rate1 * delta * hiddenLayer[i] + rate2 * lastChangeOutput[i][j];
	weightOutput[i][j] -= change;
	lastChangeOutput[i][j] = change;
      }
    }

    for (int i; i < hiddenLayer.length-1; ++i) {
      for (int j; j < inputLayer.length; ++j) {
	auto delta = errHidden[i] * dsigmoid(hiddenLayer[i]);
	auto change = rate1 * delta * inputLayer[j] + rate2 * lastChangeHidden[i][j];
	weightHidden[i][j] -= change;
	lastChangeHidden[i][j] = change;
      }
    }
  }

  void train(double[][] inputs, double[][] targets, int iteration) {
    int iter_flag = -1;

    for (int i; i < iteration; ++i) {
      
      float cur_err = 0.0;
      for (int j; j < inputs.length; ++j) {
	auto idx_arr = genRandomIdx(inputs.length);
	forward(inputs[idx_arr[j]]);
	feedback(targets[idx_arr[j]]);

	if ((j+1)%1000 == 0) {
	  if (iter_flag != i) {
	    writeln("");
	    iter_flag = i;
	  }
	  writefln("iteration %dth / progress %.2f", i+1, cast(double) j*100/cast(double) inputs.length);
	}
      }
      if ((iteration >= 10 && (i+1)%(iteration/10) == 0) || iteration < 10) {
	writefln("\niteration %dth MSE: %.5f", i+1, cur_err / cast(double) inputs.length);
      }
    }
    writeln("done.");
  }

}


void main() {
  auto nn = new NeuralNetwork(2, 3, 1, true);
  double[][] inputs = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
  double[][] targets = [[0.0], [1.0], [1.0], [2.0]];

  nn.train(inputs, targets, 1000);

  foreach (p; inputs) {
    writeln(nn.forward(p));
  }
}
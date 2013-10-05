## About

experimental Neural Network written in D.


## Usage

```
  auto nn = new NeuralNetwork(2, 3, 1, true);
  double[][] inputs = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
  double[][] targets = [[0.0], [1.0], [1.0], [2.0]];

  nn.train(inputs, targets, 1000);

  foreach (p; inputs) {
    writeln(nn.forward(p));
  }
```
import { Gaussian } from './gaussian';
import { TransferFunction } from './transfer-function';
import { TransferFunctions } from './transfer-functions';

export class BackPropagation {
  layerCount: number;
  inputSize: number;
  layerSize: number[];
  transferFunction: TransferFunction[];

  layerOutput: number[][];
  layerInput: number[][];
  bias: number[][];
  delta: number[][];
  previousBiasDelta: number[][]; // Momentum

  weight: number[][][];
  previousWeightDelta: number[][][];

  constructor(layerSizes: number[], transferFunctions: TransferFunction[]) {
    if (
      transferFunctions.length !== layerSizes.length ||
      transferFunctions[0] !== TransferFunction.NONE
    ) {
      throw new TypeError('Invalid parameters');
    }

    this.layerCount = layerSizes.length - 1;
    this.inputSize = layerSizes[0];
    this.layerSize = [];

    for (var i = 0; i < this.layerCount; i++) {
      this.layerSize[i] = layerSizes[i + 1];
    }

    this.transferFunction = [];
    for (var i = 0; i < this.layerCount; i++) {
      this.transferFunction[i] = transferFunctions[i + 1];
    }

    // Length = layerCount
    this.bias = [];
    this.delta = [];
    this.layerInput = [];
    this.layerOutput = [];
    this.previousBiasDelta = [];

    this.weight = [];
    this.previousWeightDelta = [];

    // Fill the second index (Length = layerSize)
    for (var l = 0; l < this.layerCount; l++) {
      this.bias[l] = [];
      this.delta[l] = [];
      this.layerInput[l] = [];
      this.layerOutput[l] = [];
      this.previousBiasDelta[l] = [];

      // Length = (l === 0 ? inputSize : layerSize[l - 1])
      this.weight[l] = [];
      this.previousWeightDelta[l] = [];

      // Length = layerSize[l]
      for (var i = 0; i < (l === 0 ? this.inputSize : this.layerSize[l - 1]); i++) {
        this.weight[l][i] = [];
        this.previousWeightDelta[l][i] = [];
      }
    }

    // Initialize the weights
    for (var l = 0; l < this.layerCount; l++) {
      for (var j = 0; j < this.layerSize[l]; j++) {
        this.bias[l][j] = Gaussian.getNormalGaussian();
        this.previousBiasDelta[l][j] = 0.0;
        this.layerOutput[l][j] = 0.0;
        this.layerInput[l][j] = 0.0;
        this.delta[l][j] = 0.0;
      }

      for (var i = 0; i < (l === 0 ? this.inputSize : this.layerSize[l - 1]); i++) {
        for (var j = 0; j < this.layerSize[l]; j++) {
          this.weight[l][i][j] = Gaussian.getNormalGaussian();
          this.previousWeightDelta[l][i][j] = 0.0;
        }
      }
    }
  }

  run(input: number[]) {
    // Length last layer (this.layerSize[this.layerCount - 1])
    let output: number[] = [];

    if (input.length !== this.inputSize) {
      throw new TypeError('Invalid input');
    }

    // Run the network
    for (var l = 0; l < this.layerCount; l++) {
      for (var j = 0; j < this.layerSize[l]; j++) {
        let sum = 0.0;
        for (var i = 0; i < (l === 0 ? this.inputSize : this.layerSize[l - 1]); i++) {
          sum += this.weight[l][i][j] * (l === 0 ? input[i] : this.layerOutput[l - 1][i]);
        }

        sum += this.bias[l][j];

        this.layerInput[l][j] = sum;
        this.layerOutput[l][j] = TransferFunctions.evaluate(this.transferFunction[l], sum);
      }
    }

    // Copy the output to the output array
    for (var i = 0; i < this.layerSize[this.layerCount - 1]; i++) {
      output[i] = this.layerOutput[this.layerCount - 1][i];
    }

    return output;
  }

  train(input: number[], desired: number[], trainingRate: number, momentum: number) {
    if (input.length !== this.inputSize || desired.length !== this.layerSize[this.layerCount - 1]) {
      throw new TypeError('Invalid input');
    }

    let error = 0.0;
    let sum = 0.0;
    let weightDelta = 0.0;
    let biasDelta = 0.0;
    let output: number[] = [];

    output = this.run(input);

    // Back-propagate the error
    for (var l = this.layerCount - 1; l >= 0; l--) {
      // Output layer
      if (l === this.layerCount - 1) {
        for (var k = 0; k < this.layerSize[l]; k++) {
          this.delta[l][k] = output[k] - desired[k];
          error += Math.pow(this.delta[l][k], 2);

          this.delta[l][k] *= TransferFunctions.evaluateDerivative(
            this.transferFunction[l],
            this.layerInput[l][k]
          );
        }
      } else {
        // Hidden layer
        for (var i = 0; i < this.layerSize[l]; i++) {
          sum = 0.0;
          for (var j = 0; j < this.layerSize[l + 1]; j++) {
            sum += this.weight[l + 1][i][j] * this.delta[l + 1][j];
          }

          sum *= TransferFunctions.evaluateDerivative(
            this.transferFunction[l],
            this.layerInput[l][i]
          );

          this.delta[l][i] = sum;
        }
      }
    }

    // Update the weights and biases
    for (var l = 0; l < this.layerCount; l++) {
      for (var i = 0; i < (l === 0 ? this.inputSize : this.layerSize[l - 1]); i++) {
        for (var j = 0; j < this.layerSize[l]; j++) {
          weightDelta =
            trainingRate * this.delta[l][j] * (l === 0 ? input[i] : this.layerOutput[l - 1][i]) +
            momentum * this.previousWeightDelta[l][i][j];

          this.weight[l][i][j] -= weightDelta;
          this.previousWeightDelta[l][i][j] = weightDelta;
        }
      }
    }

    for (var l = 0; l < this.layerCount; l++) {
      for (var i = 0; i < this.layerSize[l]; i++) {
        biasDelta = trainingRate * this.delta[l][i];

        this.bias[l][i] -= biasDelta + momentum * this.previousBiasDelta[l][i];
        this.previousBiasDelta[l][i] = biasDelta;
      }
    }

    return error;
  }
}

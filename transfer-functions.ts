import { TransferFunction } from './transfer-function';

export class TransferFunctions {
  static evaluate(tFunc: TransferFunction, input: number) {
    switch (tFunc) {
      case TransferFunction.SIGMOID:
        return TransferFunctions.sigmoid(input);
      case TransferFunction.LINEAR:
        return TransferFunctions.linear(input);
      case TransferFunction.GAUSSIAN:
        return TransferFunctions.gaussian(input);
      case TransferFunction.RATIONAL_SIGMOID:
        return TransferFunctions.rationalSigmoid(input);
      default:
        return 0.0;
    }
  }

  static evaluateDerivative(tFunc: TransferFunction, input: number) {
    switch (tFunc) {
      case TransferFunction.SIGMOID:
        return TransferFunctions.sigmoidDerivative(input);
      case TransferFunction.LINEAR:
        return TransferFunctions.linearDerivative(input);
      case TransferFunction.GAUSSIAN:
        return TransferFunctions.gaussianDerivative(input);
      case TransferFunction.RATIONAL_SIGMOID:
        return TransferFunctions.rationalSigmoidDerivative(input);
      default:
        return 0.0;
    }
  }

  static sigmoid(x: number) {
    return 1.0 / (1.0 + Math.exp(-x));
  }

  static sigmoidDerivative(x: number) {
    return TransferFunctions.sigmoid(x) * (1 - TransferFunctions.sigmoid(x));
  }

  static linear(x: number) {
    return x;
  }

  static linearDerivative(x: number) {
    return 1.0;
  }

  static gaussian(x: number) {
    return Math.exp(-Math.pow(x, 2));
  }

  static gaussianDerivative(x: number) {
    return -2.0 * TransferFunctions.gaussian(x) * x;
  }

  static rationalSigmoid(x: number) {
    return x / (1.0 + Math.sqrt(1.0 + x * x));
  }

  static rationalSigmoidDerivative(x: number) {
    let val = Math.sqrt(1.0 + x * x);
    return 1.0 / (val * (1 + val));
  }
}

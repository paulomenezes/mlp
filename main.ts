import { BackPropagation } from './back-propagation';
import { TransferFunction } from './transfer-function';

let network = new BackPropagation(
  [1, 2, 1],
  [TransferFunction.NONE, TransferFunction.GAUSSIAN, TransferFunction.LINEAR]
);

let input: number[] = [1.0];
let desired: number[] = [2.5];
let output: number[];

let error = 0.0;

for (var i = 0; i < 1000; i++) {
  error = network.train(input, desired, 0.15, 0.1);
  output = network.run(input);

  if (i % 100 === 0) {
    console.log(`Interation ${i}: \n\tInput ${input[0]} Output ${output[0]} Error ${error}`);
  }
}

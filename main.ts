const ChartjsNode = require('chartjs-node');

import { BackPropagation } from './back-propagation';
import { TransferFunction } from './transfer-function';

import * as fs from 'fs';

const readFile = (name: string) => {
  let input: number[][] = [];
  let output: number[][] = [];

  const data = fs.readFileSync(name).toString().split('\n');
  data.forEach(line => {
    let inputLine: number[] = [];
    line.substr(line.indexOf(',') + 1).split(',').forEach(value => {
      inputLine.push(+value);
    });

    if (inputLine.length === 16) {
      input.push(inputLine);

      let outputLine: number[] = [];
      for (var i = 0; i < 26; i++) {
        outputLine[i] = 0;
      }

      outputLine[+line.substr(0, line.indexOf(','))] = 1;
      output.push(outputLine);
    }
  });

  return [input, output];
};

const trainingData = readFile('./data/training.txt');

let input: number[][] = trainingData[0];
let output: number[][] = trainingData[1];

let network = new BackPropagation([16, 26, 26], [TransferFunction.NONE, TransferFunction.SIGMOID, TransferFunction.SIGMOID]);

const maxCount = 500;
const size = input.length;

let error = 0.0;
let count = 0;

let epochs = [];
let values = [];

do {
  count++;
  error = 0.0;

  for (var i = 0; i < size; i++) {
    error += network.train(input[i], output[i], 0.15, 0.1);
  }

  error = error / size;

  // Show progress
  epochs.push(count);
  values.push(error);
  console.log(`Epoch ${count} completed with error ${error}`);
} while (error > 0.1 && count <= maxCount);

fs.writeFile('./output/result.txt', values);

const testData = readFile('./data/test.txt');
let inputTest: number[][] = testData[0];
let outputTest: number[][] = testData[1];

let correct = 0;
let networkOutput: number[];

for (var i = 0; i < inputTest.length; i++) {
  networkOutput = network.run(inputTest[i]);
  let highest = 0;
  let index = 0;

  let rightIndex = 0;

  for (var j = 0; j < networkOutput.length; j++) {
    if (networkOutput[j] >= highest) {
      highest = networkOutput[j];
      index = j;
    }

    if (outputTest[i][j] === 1) {
      rightIndex = j;
    }
  }

  if (rightIndex === index) {
    correct++;
  }
}

console.log(correct, inputTest.length);

var chartNode = new ChartjsNode(1920, 1080);
chartNode
  .drawChart({
    type: 'line',
    data: {
      labels: epochs,
      datasets: [
        {
          label: 'Error',
          data: values,
          borderColor: 'rgb(255, 0, 0)',
          fill: false
        }
      ]
    },
    options: {
      width: 1920,
      height: 1080
    }
  })
  .then((streamResult: any) => {
    return chartNode.writeImageToFile('image/png', './output/MSE.png');
  });

/* XOR-Gate Example
let network = new BackPropagation([2, 2, 1], [TransferFunction.NONE, TransferFunction.SIGMOID, TransferFunction.LINEAR]);

let input: number[][] = [[0, 0], [0, 1], [1, 0], [1, 1]];
let output: number[][] = [[0], [1], [1], [0]];

const maxCount = 10000;
let error = 0.0;
let count = 0;

do {
  count++;
  error = 0.0;

  for (var i = 0; i < 4; i++) {
    error += network.train(input[i], output[i], 0.15, 0.1);
  }

  // Show progress
  if (count % 250 === 0) {
    console.log(`Epoch ${count} completed with error ${error}`);
  }
} while (error > 0.00001 && count <= maxCount);

let networkOutput: number[];
for (var i = 0; i < 4; i++) {
  networkOutput = network.run(input[i]);
  console.log(`Case ${i} ${input[i][0]} xor ${input[i][1]} = ${networkOutput[0]}`);
}
*/

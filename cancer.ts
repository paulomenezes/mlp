console.time('mlp');

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
    line.substr(0, line.length - 2).split(',').forEach(value => {
      inputLine.push(+value);
    });

    if (inputLine.length === 9) {
      input.push(inputLine);

      let outputLine: number[] = [];
      for (var i = 0; i < 2; i++) {
        outputLine[i] = 0;
      }

      if (+line.substr(line.length - 1, 1) === 2) {
        outputLine[0] = 1;
      } else {
        outputLine[1] = 1;
      }

      //outputLine[+line.substr(0, line.indexOf(','))] = 1;
      output.push(outputLine);
    }
  });

  return [input, output];
};

const trainingData = readFile('./data/training.breast.txt');

let valuesMean: any[] = [];

for (var index = 0; index < 30; index++) {
  let values: number[] = [];

  let input: number[][] = trainingData[0];
  let output: number[][] = trainingData[1];

  let network = new BackPropagation([9, 2, 2], [TransferFunction.NONE, TransferFunction.SIGMOID, TransferFunction.SIGMOID]);

  const maxCount = 500;
  const size = input.length;

  let error = 0.0;
  let count = 0;

  do {
    count++;
    error = 0.0;

    for (var i = 0; i < size; i++) {
      error += network.train(input[i], output[i], 0.15, 0.1);
    }

    error = error / size;

    // Show progress
    values.push(error);

    if (count % 100 === 0) {
      console.log(`Epoch ${count} completed with error ${error}`);
    }
  } while (error > 0.01 && count <= maxCount);

  fs.writeFile('./output-cancer-2/result-' + index + '.txt', values);

  const testData = readFile('./data/test.breast.txt');
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

  valuesMean.push(values);
}

let mean: number[] = [];
let labels: number[] = [];
valuesMean.forEach(d => {
  d.forEach((b: number, i: number) => {
    if (!mean[i]) mean.push(0);
    mean[i] += b;
  });
});

mean.forEach((b, i) => {
  labels.push(i);
  mean[i] = mean[i] / valuesMean.length;
});

var chartNode = new ChartjsNode(1920, 1080);
chartNode
  .drawChart({
    type: 'line',
    data: {
      labels: labels,
      datasets: [
        {
          label: 'Error',
          data: mean,
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
    return chartNode.writeImageToFile('image/png', './output-cancer-2/MSE.png');
  });

console.timeEnd('mlp');

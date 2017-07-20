export class Gaussian {
  static getRandomGaussian(mean: number, stddev: number) {
    let u, v, s, t;

    do {
      u = 2.0 * Math.random() - 1.0;
      v = 2.0 * Math.random() - 1.0;
    } while (u * u + v * v > 1 || (u === 0 && v === 0));

    s = u * u + v * v;
    t = Math.sqrt(-2.0 * Math.log(s) / s);

    return [stddev * u * t + mean, stddev * v * t + mean];
  }

  static getOneRandomGaussian(mean: number, stddev: number) {
    return Gaussian.getRandomGaussian(mean, stddev)[0];
  }

  static getNormalGaussian() {
    return Gaussian.getOneRandomGaussian(0.0, 1.0);
  }
}

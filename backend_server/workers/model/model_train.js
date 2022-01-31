const Queue = require('bull');
const config = require('config');
const axios = require('axios');

const modelTrainQueue = new Queue('model-train-queue', {
  redis: {
    port: config.get('redisPort'),
    host: config.get('redisIP'),
    password: config.get('redisPassword'),
  },
});

modelTrainQueue.process(async (job, done) => {
  try {
    const res = await axios.post(
      `http://${config.get('loadBalancerIP')}:${config.get(
        'loadBalancerPort'
      )}/model/train`,
      JSON.stringify(job.data.body),
      { headers: job.data.headers, timeout: 1000 * 60 * 10 }
    );

    done(null, { resData: res.data });
  } catch (err) {
    return done(new Error(err));
  }
});

module.exports = { modelTrainQueue };

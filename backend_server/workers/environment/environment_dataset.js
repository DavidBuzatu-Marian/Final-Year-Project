const Queue = require('bull');
const config = require('config');
const axios = require('axios');

const environmentDatasetQueue = new Queue('environment-dataset-queue', {
  redis: {
    port: config.get('redisPort'),
    host: config.get('redisIP'),
    password: config.get('redisPassword'),
  },
});

environmentDatasetQueue.process(async (job, done) => {
  let res;
  try {
    res = await axios.post(
      `http://${config.get('loadBalancerIP')}:${config.get(
        'loadBalancerPort'
      )}/environment/dataset${job.data.endpoint}`,
      JSON.stringify(job.data.body),
      {
        headers: job.data.headers,
        timeout: 1000 * 60 * 10,
      }
    );
    return done(null, { resData: res.data });
  } catch (error) {
    return done(new Error(error));
  }
});

module.exports = { environmentDatasetQueue };

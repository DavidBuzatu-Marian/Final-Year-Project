const Queue = require('bull');
const config = require('config');
const axios = require('axios');

const environmentCreateQueue = new Queue('environment-create-queue', {
  redis: {
    port: config.get('redisPort'),
    host: config.get('redisIP'),
    password: config.get('redisPassword'),
  },
});

environmentCreateQueue.process(async (job) => {
  const res = await axios.post(
    `http://${config.get('loadBalancerIP')}:${config.get(
      'loadBalancerPort'
    )}/environment/create`,
    JSON.stringify(job.data.body),
    { headers: job.data.headers, timeout: 1000 * 60 * 10 }
  );

  done(null, res.data);
});

module.exports = { environmentCreateQueue };

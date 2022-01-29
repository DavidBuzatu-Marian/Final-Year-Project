const Queue = require('bull');
const config = require('config');
const axios = require('axios');

const environmentDeleteQueue = new Queue('environment-delete-queue', {
  redis: {
    port: config.get('redisPort'),
    host: config.get('redisIP'),
    password: config.get('redisPassword'),
  },
});

environmentDeleteQueue.process(async (job, done) => {
  const res = await axios.delete(
    `http://${config.get('loadBalancerIP')}:${config.get(
      'loadBalancerPort'
    )}/environment/delete`,
    {
      headers: job.data.headers,
      timeout: 1000 * 60 * 10,
      data: JSON.stringify(job.data.body),
    }
  );

  done(null, { resData: res.data });
});

module.exports = { environmentDeleteQueue };

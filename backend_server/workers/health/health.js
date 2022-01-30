const Queue = require('bull');
const config = require('config');
const axios = require('axios');

const healthQueue = new Queue('health-queue', {
  redis: {
    port: config.get('redisPort'),
    host: config.get('redisIP'),
    password: config.get('redisPassword'),
  },
});

healthQueue.process(async (job, done) => {
  try {
    const res = await axios.get(
      `http://${config.get('loadBalancerIP')}:${config.get(
        'loadBalancerPort'
      )}/health/status`,
      { headers: job.data.headers }
    );

    done(null, { data: res.data });
  } catch (err) {
    return done(new Error(err));
  }
});

module.exports = { healthQueue };

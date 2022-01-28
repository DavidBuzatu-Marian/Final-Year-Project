const Queue = require('bull');
const config = require('config');

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
    job.data.body,
    { headers: job.data.headers }
  );
  done(null, res.data);
});

module.exports = { environmentCreateQueue };

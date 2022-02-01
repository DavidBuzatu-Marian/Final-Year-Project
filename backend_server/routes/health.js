const express = require('express');
const { removeUrlParams } = require('../hooks/url');
const { healthQueue } = require('../workers/health/health');
const { strategyMap, QueueStrategy } = require('../workers/strategy');
const router = express.Router();

router.get('/status', async (req, res) => {
  if (!strategyMap.has(req.originalUrl)) {
    return res.statusCode(404).send('Endpoint was not found');
  }
  const routeStrategy = new QueueStrategy();
  routeStrategy.setStrategy(strategyMap.get(req.originalUrl));
  delete req.headers['content-length'];
  const job_headers = { ...req.headers, 'content-type': 'application/json' };
  const job = await routeStrategy.add({
    headers: job_headers,
  });
  return res.status(202).json({ jobLink: `/api/health/status/${job.id}` });
});

router.get('/status/:id', async (req, res) => {
  const url = removeUrlParams(req.originalUrl);
  if (!strategyMap.has(url)) {
    return res.status(404).send('Endpoint not found');
  }

  const routeStrategy = new QueueStrategy();
  routeStrategy.setStrategy(strategyMap.get(url));
  const id = req.params.id;
  const job = await healthQueue.getJob(id);
  if (job === null) {
    return res.status(400).end();
  } else {
    const jobState = await job.getState();
    const jobFailReason = job.failedReason;
    let jobResult;
    if (jobState === 'completed') {
      jobResult = await job.finished();
    }
    return res.json({ id, jobState, jobFailReason, jobResult });
  }
});

module.exports = router;

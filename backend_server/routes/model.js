const express = require('express');
const router = express.Router();
const { strategyMap, QueueStrategy } = require('../workers/strategy');

const {
  handleJobResponse,
  createJobBody,
  createJobHeader,
} = require('../hooks/environment');
const { removeUrlParams } = require('../hooks/url');

router.post('/*', async (req, res) => {
  if (!strategyMap.has(req.originalUrl)) {
    return res.statusCode(404).send('Endpoint not found');
  }
  const routeStrategy = new QueueStrategy();
  routeStrategy.setStrategy(strategyMap.get(req.originalUrl));
  delete req.headers['content-length'];
  const jobHeader = createJobHeader(req, 'application/json');
  const jobBody = createJobBody(req);
  const job = await routeStrategy.add({
    headers: jobHeader,
    body: jobBody,
  });
  return res.status(202).json({ jobLink: `/api/model/create/${job.id}` });
});

router.get('/*:id', async (req, res) => {
  let url = removeUrlParams(req.originalUrl);

  if (!strategyMap.has(url)) {
    return res.status(404).send('Endpoint not found');
  }
  const routeStrategy = new QueueStrategy();
  routeStrategy.setStrategy(strategyMap.get(url));
  const id = req.params.id;
  const job = await routeStrategy.getJob(id);
  return await handleJobResponse(res, id, job);
});

module.exports = router;

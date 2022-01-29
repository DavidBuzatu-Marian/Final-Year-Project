const express = require('express');
const { environmentCreateQueue } = require('../workers/environment_create');
const router = express.Router();

router.post('/create', async (req, res) => {
  delete req.headers['content-length'];
  const job_headers = { ...req.headers, 'content-type': 'application/json' };
  const job_body = { ...req.body, user_id: req.headers['x-auth'] };
  const job = await environmentCreateQueue.add({
    headers: job_headers,
    body: job_body,
  });
  return res.status(202).json({ jobLink: `/api/environment/create/${job.id}` });
});

router.get('/create/:id', async (req, res) => {
  const id = req.params.id;
  const job = await environmentCreateQueue.getJob(id);
  if (job === null) {
    return res.status(400).end();
  } else {
    const jobState = await job.getState();
    const jobFailReason = job.failedReason;
    return res.json({ id, jobState, jobFailReason });
  }
});

module.exports = router;

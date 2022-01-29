const express = require('express');
const {
  environmentCreateQueue,
} = require('../workers/environment/environment_create');
const {
  environmentDeleteQueue,
} = require('../workers/environment/environment_delete');
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

router.delete('/delete', async (req, res) => {
  delete req.headers['content-length'];
  const job_headers = { ...req.headers, 'content-type': 'application/json' };
  const job_body = { ...req.body, user_id: req.headers['x-auth'] };
  const job = await environmentDeleteQueue.add({
    headers: job_headers,
    body: job_body,
  });
  return res.status(202).json({ jobLink: `/api/environment/delete/${job.id}` });
});

router.get('/delete/:id', async (req, res) => {
  const id = req.params.id;
  const job = await environmentDeleteQueue.getJob(id);
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

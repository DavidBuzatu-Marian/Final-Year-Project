const express = require('express');
const {
  handleJobResponse,
  createJobBody,
  createJobHeader,
} = require('../hooks/environment');
const {
  environmentCreateQueue,
} = require('../workers/environment/environment_create');
const {
  environmentDatasetQueue,
} = require('../workers/environment/environment_dataset');
const {
  environmentDeleteQueue,
} = require('../workers/environment/environment_delete');
const router = express.Router();

router.post('/create', async (req, res) => {
  delete req.headers['content-length'];
  const job_headers = createJobHeader(req, 'application/json');
  const job_body = createJobBody(req);
  const job = await environmentCreateQueue.add({
    headers: job_headers,
    body: job_body,
  });
  return res.status(202).json({ jobLink: `/api/environment/create/${job.id}` });
});

router.get('/create/:id', async (req, res) => {
  const id = req.params.id;
  const job = await environmentCreateQueue.getJob(id);
  return await handleJobResponse(res, id, job);
});

router.delete('/delete', async (req, res) => {
  delete req.headers['content-length'];
  const job_headers = createJobHeader(req, 'application/json');
  const job_body = createJobBody(req);
  const job = await environmentDeleteQueue.add({
    headers: job_headers,
    body: job_body,
  });
  return res.status(202).json({ jobLink: `/api/environment/delete/${job.id}` });
});

router.get('/delete/:id', async (req, res) => {
  const id = req.params.id;
  const job = await environmentDeleteQueue.getJob(id);
  return await handleJobResponse(res, id, job);
});

router.post('/dataset/data', async (req, res) => {
  delete req.headers['content-length'];
  const job_headers = createJobHeader(req, 'multipart/form-data');
  const job_body = createJobBody(req);
  const job = await environmentDatasetQueue.add({
    headers: job_headers,
    body: job_body,
    endpoint: '/data',
  });
  return res
    .status(202)
    .json({ jobLink: `/api/environment/dataset/${job.id}` });
});

router.get('/dataset/:id', async (req, res) => {
  const id = req.params.id;
  const job = await environmentDatasetQueue.getJob(id);
  return await handleJobResponse(res, id, job);
});

module.exports = router;

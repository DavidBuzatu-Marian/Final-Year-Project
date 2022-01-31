const express = require('express');
const router = express.Router();
const { modelCreateQueue } = require('../workers/model/model_create');

const {
  handleJobResponse,
  createJobBody,
  createJobHeader,
} = require('../hooks/environment');

router.post('/create', async (req, res) => {
  delete req.headers['content-length'];
  const jobHeader = createJobHeader(req, 'application/json');
  const jobBody = createJobBody(req);
  const job = await modelCreateQueue.add({
    headers: jobHeader,
    body: jobBody,
  });
  return res.status(202).json({ jobLink: `/api/model/create/${job.id}` });
});

router.get('/create/:id', async (req, res) => {
  const id = req.params.id;
  const job = await modelCreateQueue.getJob(id);
  return await handleJobResponse(res, id, job);
});

module.exports = router;

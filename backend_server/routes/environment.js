const express = require('express');
const multer = require('multer');
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

const storage = multer.diskStorage({
  destination: './temp/',
  filename: (_, file, cb) => {
    cb(null, file.originalname);
  },
});
const upload = multer({ storage: storage });

router.post('/create', async (req, res) => {
  delete req.headers['content-length'];
  const jobHeader = createJobHeader(req, 'application/json');
  const jobBody = createJobBody(req);
  const job = await environmentCreateQueue.add({
    headers: jobHeader,
    body: jobBody,
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
  const jobHeader = createJobHeader(req, 'application/json');
  const jobBody = createJobBody(req);
  const job = await environmentDeleteQueue.add({
    headers: jobHeader,
    body: jobBody,
  });
  return res.status(202).json({ jobLink: `/api/environment/delete/${job.id}` });
});

router.get('/delete/:id', async (req, res) => {
  const id = req.params.id;
  const job = await environmentDeleteQueue.getJob(id);
  return await handleJobResponse(res, id, job);
});

router.post(
  '/dataset/data',
  upload.fields([
    { name: 'train_data', maxCount: 10000 },
    { name: 'train_labels', maxCount: 10000 },
  ]),
  async (req, res) => {
    delete req.headers['content-length'];
    const jobHeader = createJobHeader(req, 'multipart/form-data');
    const jobBody = createJobBody(req);
    const job = await environmentDatasetQueue.add({
      headers: jobHeader,
      body: jobBody,
      endpoint: `/data?user_id=${req.query.user_id}&environment_id=${req.query.environment_id}`,
    });
    return res
      .status(202)
      .json({ jobLink: `/api/environment/dataset/${job.id}` });
  }
);

router.post('/dataset/distribution', async (req, res) => {
  delete req.headers['content-length'];
  const jobHeader = createJobHeader(req, 'application/json');
  const jobBody = createJobBody(req);
  const job = await environmentDatasetQueue.add({
    headers: jobHeader,
    body: jobBody,
    endpoint: '/distribution',
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

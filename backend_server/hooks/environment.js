const handleJobResponse = async (res, id, job) => {
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
};

const createJobBody = (req) => {
  return { ...req.body, user_id: req.headers['x-auth'] };
};

const createJobHeader = (req, contentType) => {
  return { ...req.headers, 'content-type': contentType };
};

module.exports = { handleJobResponse, createJobBody, createJobHeader };

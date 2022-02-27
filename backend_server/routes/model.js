const express = require("express");
const router = express.Router();
const { strategyMap, QueueStrategy } = require("../workers/strategy");
const path = require("path");

const {
  handleJobResponse,
  createJobBody,
  createJobHeader,
} = require("../hooks/environment");
const { removeUrlParams } = require("../hooks/url");

router.post("/*", async (req, res) => {
  if (!strategyMap.has(req.originalUrl)) {
    return res.statusCode(404).send("Endpoint not found");
  }
  const routeStrategy = new QueueStrategy();
  routeStrategy.setStrategy(strategyMap.get(req.originalUrl));
  delete req.headers["content-length"];
  const jobHeader = createJobHeader(req, "application/json");
  const jobBody = createJobBody(req);
  const job = await routeStrategy.add({
    headers: jobHeader,
    body: jobBody,
  });
  return res.status(202).json({ jobLink: `${req.originalUrl}/${job.id}` });
});

router.get("/download", async (req, res) => {
  if (!req.body || (req.body && !req.body.environment_id)) {
    return res.status(400).send("Environment id is required");
  }
  environment_id = req.body.environment_id;
  res.download(
    path.join(
      __dirname,
      "..",
      "workers",
      "model",
      "models",
      `${environment_id}.pth`
    )
  );
});

router.get("/*:id", async (req, res) => {
  const url = removeUrlParams(req.originalUrl);
  if (!strategyMap.has(url)) {
    return res.status(404).send("Endpoint not found");
  }

  const routeStrategy = new QueueStrategy();
  routeStrategy.setStrategy(strategyMap.get(url));
  const id = req.params.id;
  const job = await routeStrategy.getJob(id);
  return await handleJobResponse(res, id, job);
});

module.exports = router;

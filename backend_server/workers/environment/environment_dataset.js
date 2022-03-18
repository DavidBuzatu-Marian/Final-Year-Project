const Queue = require("bull");
const config = require("config");
const axios = require("axios");
const fs = require("fs");
const { deleteLocalFiles } = require("../../hooks/upload");
const FormData = require("form-data");

const environmentDatasetQueue = new Queue("environment-dataset-queue", {
  redis: {
    port: config.get("redisPort"),
    host: config.get("redisIP"),
    password: config.get("redisPassword"),
  },
});

environmentDatasetQueue.process(async (job, done) => {
  try {
    const res = await makeAxiosRequest(job);
    return done(null, { resData: res.data });
  } catch (error) {
    return done(new Error(error));
  } finally {
    if (job.data.body.files) {
      deleteLocalFiles(job.data.body);
    }
  }
});

const makeAxiosRequest = async (job) => {
  if (job.data.body.files) {
    const reqFiles = job.data.body.files;
    let formData = new FormData();
    for (key in reqFiles) {
      for (file of reqFiles[key]) {
        formData.append(key, fs.createReadStream(file.path));
      }
    }
    return await axios.post(
      `http://${config.get("loadBalancerIP")}:${config.get(
        "loadBalancerPort"
      )}/environment/dataset${job.data.endpoint}`,
      formData,
      {
        headers: formData.getHeaders(),
        timeout: 1000 * 60 * 10,
      }
    );
  }
  return await axios.post(
    `http://${config.get("loadBalancerIP")}:${config.get(
      "loadBalancerPort"
    )}/environment/dataset${job.data.endpoint}`,
    JSON.stringify(job.data.body),
    {
      headers: job.data.headers,
      timeout: 1000 * 60 * 10,
    }
  );
};

module.exports = { environmentDatasetQueue };

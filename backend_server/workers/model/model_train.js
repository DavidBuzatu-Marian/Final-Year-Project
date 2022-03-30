const Queue = require("bull");
const config = require("config");
const axios = require("axios");
const fs = require("fs");
const path = require("path");

const modelTrainQueue = new Queue("model-train-queue", {
  redis: {
    port: config.get("redisPort"),
    host: config.get("redisIP"),
    password: config.get("redisPassword"),
  },
});

modelTrainQueue.process(async (job, done) => {
  try {
    const res = await axios.post(
      `http://${config.get("loadBalancerIP")}:${config.get(
        "loadBalancerPort"
      )}/model/train`,
      JSON.stringify(job.data.body),
      { headers: job.data.headers, timeout: 1000 * 60 * 40 }
    );
    fs.writeFileSync(
      path.join(__dirname, "models", `${job.data.body.environment_id}.pth`),
      res.data
    );
    done(null, { resData: res.data });
  } catch (err) {
    return done(new Error(err));
  }
});

module.exports = { modelTrainQueue };

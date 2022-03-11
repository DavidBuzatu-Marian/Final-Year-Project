const Queue = require("bull");
const config = require("config");
const axios = require("axios");

const modelCreateQueue = new Queue("model-create-queue", {
  redis: {
    port: config.get("redisPort"),
    host: config.get("redisIP"),
    password: config.get("redisPassword"),
  },
});

modelCreateQueue.process(async (job, done) => {
  try {
    const res = await axios.post(
      `http://${config.get("loadBalancerIP")}:${config.get(
        "loadBalancerPort"
      )}/model/create`,
      JSON.stringify(job.data.body),
      { headers: job.data.headers, timeout: 1000 * 60 * 10 }
    );

    done(null, { resData: res.data });
  } catch (err) {
    console.log(err);
    return done(new Error(err));
  }
});

module.exports = { modelCreateQueue };

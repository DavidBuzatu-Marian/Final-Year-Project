const { modelCreateQueue } = require('./model/model_create');
const { modelTrainQueue } = require('./model/model_train');

class QueueStrategy {
  setStrategy = (queue) => {
    this.queue = queue;
  };

  add = async (job) => {
    return await this.queue.add(job);
  };

  getJob = async (jobId) => {
    return await this.queue.getJob(jobId);
  };
}

const strategyMap = new Map();
strategyMap.set('/api/model/create', modelCreateQueue);
strategyMap.set('/api/model/train', modelTrainQueue);

module.exports = { strategyMap, QueueStrategy };

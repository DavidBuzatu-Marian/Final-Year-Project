const mongoose = require('mongoose');

const EnvironmentTrainingDataDistributionSchema = new mongoose.Schema(
  {
    user_id: {
      type: mongoose.Types.ObjectId,
      ref: 'users',
      required: true,
    },
    environment_id: {
      type: mongoose.Types.ObjectId,
      ref: 'environmentsAddresses',
      required: true,
    },
    distributions: {
      type: Object,
      required: true,
    },
  },
  { collection: 'environmentsTrainingDataDistribution' }
);

module.exports = EnvironmentTrainingDataDistribution = mongoose.model(
  'environmentsTrainingDataDistribution',
  EnvironmentTrainingDataDistributionSchema,
  'environmentsTrainingDataDistribution'
);

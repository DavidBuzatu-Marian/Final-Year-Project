const mongoose = require('mongoose');

const EnvironmentDataDistributionSchema = new mongoose.Schema(
  {
    user_id: {
      type: mongoose.Types.ObjectId,
      ref: 'users',
    },
    environment_id: {
      type: mongoose.Types.ObjectId,
      ref: 'environmentsAddresses',
    },
    test_data_distribution: {
      type: [Object],
      required: true,
    },
    test_labels_data_distribution: {
      type: [Object],
      required: true,
    },
    train_data_distribution: {
      type: [Object],
      required: true,
    },
    train_labels_data_distribution: {
      type: [Object],
      required: true,
    },
    validation_data_distribution: {
      type: [Object],
      required: true,
    },
    validation_labels_data_distribution: {
      type: [Object],
      required: true,
    },
  },
  { collection: 'environmentsDataDistribution' }
);

module.exports = EnvironmentDataDistribution = mongoose.model(
  'environmentsDataDistribution',
  EnvironmentDataDistributionSchema,
  'environmentsDataDistribution'
);

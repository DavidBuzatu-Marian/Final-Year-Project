const mongoose = require('mongoose');

const EnvironmentAddressesSchema = new mongoose.Schema(
  {
    user_id: {
      type: String,
      ref: 'users',
    },
    date: {
      type: Number,
      default: Date.now,
    },
    environment_ips: {
      type: [String],
      required: true,
    },
    environment_options: {
      type: [Object],
    },
    status: {
      type: String,
    },
    machine_type: {
      type: String,
    },
  },
  { collection: 'environmentsAddresses' }
);

module.exports = EnvironmentAddresses = mongoose.model(
  'environmentsAddresses',
  EnvironmentAddressesSchema,
  'environmentsAddresses'
);

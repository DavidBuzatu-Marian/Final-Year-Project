const mongoose = require("mongoose");

const EnvironmentTrainingLogsSchema = new mongoose.Schema(
  {
    user_id: {
      type: mongoose.Types.ObjectId,
      ref: "users",
      required: true,
    },
    environment_id: {
      type: mongoose.Types.ObjectId,
      ref: "environmentsAddresses",
      required: true,
    },
    train_logs: {
      type: [Object],
      required: true,
    },
  },
  { collection: "environmentsLogs" }
);

module.exports = EnvironmentTrainingLogs = mongoose.model(
  "environmentsLogs",
  EnvironmentTrainingLogsSchema,
  "environmentsLogs"
);

const express = require('express');
const EnvironmentDataDistribution = require('../models/EnvironmentDataDistribution');
const EnvironmentTrainingDataDistribution = require('../models/EnvironmentTrainingDataDistribution');
const router = express.Router();
const mongoose = require('mongoose');

router.get('/distribution', async (req, res) => {
  const userId = req.headers['x-auth'];
  const environmentsDataDistribution = await EnvironmentDataDistribution.find({
    user_id: mongoose.mongo.ObjectId(userId),
  });
  return res.send(environmentsDataDistribution);
});

router.get('/training/distribution', async (req, res) => {
  const userId = req.headers['x-auth'];
  const environmentsTrainingDataDistribution =
    await EnvironmentTrainingDataDistribution.find({
      user_id: mongoose.mongo.ObjectId(userId),
    });
  return res.send(environmentsTrainingDataDistribution);
});

module.exports = router;

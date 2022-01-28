const express = require('express');
const router = express.Router();
const axios = require('axios');

router.get('/status', async (req, res) => {
  const health_res = await axios.get(
    'http://host.docker.internal:3000/health/status',
    { headers: req.headers }
  );
  res.json({ user_id: req.headers['x-auth'], resp: health_res.data });
});

module.exports = router;

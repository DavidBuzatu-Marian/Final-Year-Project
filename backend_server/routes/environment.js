const express = require('express');
const router = express.Router();

router.post('/create', async (req, res) => {
  res.json({ user_id: req.headers['x-auth'] });
});

module.exports = router;

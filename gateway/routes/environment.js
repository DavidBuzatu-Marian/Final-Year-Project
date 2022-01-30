const express = require('express');
const router = express.Router();
const proxy = require('express-http-proxy');
const config = require('config');

router.post('/dataset/data', proxy(`http://${config.get('backendIP')}:5005`));

module.exports = router;

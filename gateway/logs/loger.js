const morgan = require('morgan');
const path = require('path');
const rfs = require('rotating-file-stream');

// As described in the official documentation: https://github.com/expressjs/morgan
const logFileStream = rfs.createStream('general_logs.log', {
  interval: '1d',
  path: path.join(__dirname, 'logs'),
});

const registerLoggerInApp = (app) => {
  app.use(morgan('combined', { stream: logFileStream }));
};

module.exports = { registerLoggerInApp };

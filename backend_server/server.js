const express = require('express');
const bodyParser = require('body-parser');
const config = require('config');
const { connectToMongoDB } = require('./config/mongo');
const { registerLoggerInApp } = require('./logs/loger');
const app = express();

// Connect to mongo
connectToMongoDB();

// Set-up morgan
registerLoggerInApp(app);

// Set-up json and form data parser
app.use(bodyParser.urlencoded({ extended: true }));
app.use(express.json());

// Routes
app.use('/api/environment', require('./routes/environment'));
app.use('/api/health', require('./routes/health'));
// Start server
const PORT = config.get('port') || 5005;
app.listen(PORT, (err) => {
  if (err) {
    console.log(`Error: ${err}`);
  }
  console.log(`Listening on port: ${PORT}`);
});

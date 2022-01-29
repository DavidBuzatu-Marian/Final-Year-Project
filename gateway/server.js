const express = require('express');
const bodyParser = require('body-parser');
const session = require('express-session');
const config = require('config');
const passport = require('passport');
const User = require('./models/User');
const app = express();
const { connectToMongoDB } = require('./config/mongo');
const { registerLoggerInApp } = require('./logs/loger');
const { configureProxyWithApplication } = require('./routes/proxy');
const fileUpload = require('express-fileupload');
const ROUTES = require('./routes/routes');
// Connect to Mongo
connectToMongoDB();

// Set-up logger
registerLoggerInApp(app);

// Set-up session
app.use(
  session({
    secret: config.get('sessionSecret'),
    resave: false,
    saveUninitialized: true,
    cookie: { maxAge: 60 * 60 * 1000 * 12 }, // 12 hours
  })
);

// Set-up json and form data parser
app.use(bodyParser.urlencoded({ extended: true }));
app.use(express.json());
app.use(fileUpload());

// Set-up passport
app.use(passport.initialize());
app.use(passport.session());
passport.use(User.createStrategy());
passport.serializeUser(User.serializeUser());
passport.deserializeUser(User.deserializeUser());

// Routes
app.use('/api/auth', require('./routes/auth'));

// Proxy
configureProxyWithApplication(app, ROUTES);
// Start server
const PORT = config.get('port') || 5002;
app.listen(PORT, (err) => {
  if (err) {
    console.log(`Error: ${err}`);
  }
  console.log(`Listening on port: ${PORT}`);
});

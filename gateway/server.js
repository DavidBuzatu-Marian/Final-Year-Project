const express = require("express");
const session = require("express-session");
const config = require("config");
const passport = require("passport");
const User = require("./models/User");
const app = express();
const { connectToMongoDB } = require("./config/mongo");
const { registerLoggerInApp } = require("./logs/loger");
const { configureProxyWithApplication } = require("./routes/proxy");
const ROUTES = require("./routes/routes");
const cors = require("cors");
// Connect to Mongo
connectToMongoDB();

// Set-up logger
registerLoggerInApp(app);

//CORS
app.use(
  cors({
    credentials: true,
    origin: config.get("originUrl"),
    exposedHeaders: ["set-cookie"],
  })
);

// Set-up session
app.use(
  session({
    secret: config.get("sessionSecret"),
    resave: false,
    saveUninitialized: true,
    cookie: { maxAge: 60 * 60 * 1000 * 12 }, // 12 hours
  })
);

// Set-up passport
app.use(passport.initialize());
app.use(passport.session());
passport.use(User.createStrategy());
passport.serializeUser(User.serializeUser());
passport.deserializeUser(User.deserializeUser());

app.use(express.json());
// Routes
app.use("/api/environment", require("./routes/environment"));
app.use("/api/auth", require("./routes/auth"));

// Proxy
configureProxyWithApplication(app, ROUTES);

// Start server
const PORT = config.get("port") || 5002;
app.listen(PORT, (err) => {
  if (err) {
    console.log(`Error: ${err}`);
  }
  console.log(`Listening on port: ${PORT}`);
});

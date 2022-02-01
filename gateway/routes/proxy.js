const { createProxyMiddleware } = require('http-proxy-middleware');
const { ensureAuthenticated } = require('../middleware/auth');

const configureProxyWithApplication = (app, routes) => {
  routes.forEach((route) => {
    if (route.authenticated) {
      app.use(
        ensureAuthenticated,
        createProxyMiddleware('/backend', route.proxy)
      );
    } else {
      app.use(createProxyMiddleware('/backend', route.proxy));
    }
  });
};

module.exports = { configureProxyWithApplication };

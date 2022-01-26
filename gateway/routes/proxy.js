const { createProxyMiddleware } = require('http-proxy-middleware');

const configureProxyWithApplication = (app, routes) => {
  routes.forEach((route) => {
    app.use(createProxyMiddleware('/backend', route.proxy));
  });
};

module.exports = { configureProxyWithApplication };

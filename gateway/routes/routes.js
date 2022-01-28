const { fixRequestBody } = require('http-proxy-middleware');

const ROUTES = [
  {
    authenticated: true,
    proxy: {
      target: 'http://localhost:5005',
      pathRewrite: { '^/backend': '' },
      onProxyReq: (proxyReq, req) => {
        proxyReq.setHeader('x-auth', req.user._id);
        fixRequestBody(proxyReq, req);
      },
    },
  },
];

module.exports = ROUTES;

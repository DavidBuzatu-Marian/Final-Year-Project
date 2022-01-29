const { fixRequestBody } = require('http-proxy-middleware');
const config = require('config');

const ROUTES = [
  {
    authenticated: true,
    proxy: {
      target: `http://${config.get('backendIP')}:5005`,
      pathRewrite: { '^/backend': '' },
      onProxyReq: (proxyReq, req) => {
        proxyReq.setHeader('x-auth', req.user._id);
        fixRequestBody(proxyReq, req);
        proxyReq.write(JSON.stringify(req.files));
      },
    },
  },
];

module.exports = ROUTES;

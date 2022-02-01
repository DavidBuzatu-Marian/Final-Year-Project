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
        // if (req.headers.hasOwnProperty('content-length')) {
        //   proxyReq.setHeader('content-length', req.headers['content-length']);
        // }
        if (req.files !== undefined) {
          proxyReq.files = req.files;
        }
      },
    },
  },
];

module.exports = ROUTES;

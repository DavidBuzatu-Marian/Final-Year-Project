const ROUTES = [
  {
    authenticated: true,
    proxy: {
      target: 'http://localhost:5005',
      pathRewrite: { '^/backend': '' },
      onProxyReq: (proxyReq, req) => {
        proxyReq.setHeader('x-auth', req.user._id);
      },
    },
  },
];

module.exports = ROUTES;

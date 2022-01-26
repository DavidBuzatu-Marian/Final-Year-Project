const ROUTES = [
  {
    proxy: {
      target: 'http://localhost:3000',
      changeOrigin: true,
      pathRewrite: { '^/backend': '' },
    },
  },
];

module.exports = ROUTES;

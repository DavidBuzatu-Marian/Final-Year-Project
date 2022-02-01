const removeUrlParams = (url) => {
  let parsedUrl = url.split('/');
  parsedUrl.pop();
  parsedUrl = parsedUrl.join('/');
  return parsedUrl;
};

module.exports = { removeUrlParams };

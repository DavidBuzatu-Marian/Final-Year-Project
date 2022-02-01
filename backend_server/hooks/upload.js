const fs = require('fs');
const { promisify } = require('util');
const unlinkAsync = promisify(fs.unlink);

const deleteLocalFiles = async (req) => {
  for (key in req.files) {
    for (file of req.files[key]) {
      await unlinkAsync(file.path);
    }
  }
};

module.exports = { deleteLocalFiles };

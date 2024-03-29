const fs = require("fs");
const { promisify } = require("util");
const unlinkAsync = promisify(fs.unlink);

const deleteLocalFiles = async (req) => {
  for (key in req.files) {
    for (file of req.files[key]) {
      try {
        if (fs.existsSync(file.path)) {
          await unlinkAsync(file.path);
        }
      } catch (err) {
        console.log(err);
      }
    }
  }
};

module.exports = { deleteLocalFiles };

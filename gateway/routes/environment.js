const express = require('express');
const router = express.Router();
const axios = require('axios');
const config = require('config');
const FormData = require('form-data');
const multer = require('multer');
const upload = multer({ dest: 'temp/' });
const fs = require('fs');
const { deleteLocalFiles } = require('../hooks/upload');
const { ensureAuthenticated } = require('../middleware/auth');
router.post(
  '/dataset/data',
  [
    ensureAuthenticated,
    upload.fields([
      { name: 'train_data', maxCount: 10000 },
      { name: 'train_labels', maxCount: 10000 },
    ]),
  ],
  async (req, res) => {
    try {
      let formData = new FormData();
      if (!req.files) {
        return res.status(400).send('At least a file is required');
      }
      for (key in req.files) {
        for (file of req.files[key]) {
          formData.append(key, fs.createReadStream(file.path));
        }
      }
      //   console.log(formData);
      const axiosRes = await axios.post(
        `http://${config.get(
          'backendIP'
        )}:5005/api/environment/dataset/data?user_id=${
          req.query.user_id
        }&environment_id=${req.query.environment_id}`,
        formData,
        {
          headers: formData.getHeaders(),
        }
      );
      return res.json(axiosRes.data);
    } catch (error) {
      console.log(error);
      res.sendStatus(500);
    } finally {
      deleteLocalFiles(req);
    }
  }
);

module.exports = router;

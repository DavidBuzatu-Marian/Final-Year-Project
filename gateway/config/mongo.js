const mongoose = require('mongoose');
const config = require('config');

const db = config.get('mongoURI');

const connectToMongoDB = async () => {
  try {
    await mongoose.connect(db, {
      useNewUrlParser: true,
      useUnifiedTopology: true,
    });
    console.log('Connected to Mongo');
  } catch (err) {
    console.error(err.message);
    process.exit(1);
  }
};

module.exports = { connectToMongoDB };

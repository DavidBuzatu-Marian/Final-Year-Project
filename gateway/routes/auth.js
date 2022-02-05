const express = require('express');
const router = express.Router();
const passport = require('passport');
const { ensureAuthenticated } = require('../middleware/auth');

router.post('/login', passport.authenticate('local'), (req, res) => {
  res.json({ user_id: req.user._id, email: req.user.email });
});

router.get('/authenticated', ensureAuthenticated, (req, res) => {
  res.json({ user_id: req.user._id, email: req.user.email });
});

router.post('/register', (req, res) => {
  User.register(
    new User({ email: req.body.email, username: req.body.email }),
    req.body.password,
    (err, user) => {
      if (err) {
        console.log(err);
        return res.status(500).send(`Internal server error: ${err}`);
      }
      res.json('User successfully registered');
    }
  );
});

router.post('/logout', (req, res) => {
  req.logout();
  res.send('User logged out');
});

module.exports = router;

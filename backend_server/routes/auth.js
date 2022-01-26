const express = require('express');
const router = express.Router();
const passport = require('passport');
const { ensureAuthenticated } = require('../middleware/auth');

router.post('/login', passport.authenticate('local'), (req, res) => {
  console.log(req.user);
  res.send('Logged in');
});

router.get('/secret', ensureAuthenticated, (req, res) => {
  res.json({ user_id: req.user._id });
});

router.post('/register', (req, res) => {
  User.register(
    new User({ email: req.body.email }),
    req.body.password,
    (err, user) => {
      if (err) {
        console.log(err);
        return res.status(500).send(`Internal server error: ${err}`);
      }
      passport.authenticate('local')(req, res, () => {
        res.send('Account created successfully');
      });
    }
  );
});

router.post('/logout', (req, res) => {
  req.logout();
});

module.exports = router;

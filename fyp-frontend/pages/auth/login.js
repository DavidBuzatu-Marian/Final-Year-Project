import React from 'react';
import Card from '@mui/material/Card';
import CardActions from '@mui/material/CardActions';
import CardContent from '@mui/material/CardContent';
import Button from '@mui/material/Button';
import Typography from '@mui/material/Typography';
import Container from '@mui/material/Container';
import TextField from '@mui/material/TextField';
import LoginForm from '../../components/auth/loginForm';

const login = () => {
  return (
    <section>
      <Container
        maxWidth='sm'
        sx={{
          minHeight: '100vh',
          display: 'flex',
          justifyContent: 'center',
          alignItems: 'center',
        }}
      >
        <Card
          sx={{
            mx: 'auto',
            my: 2,
            minWidth: '100%',
            boxShadow: 3,
          }}
        >
          <CardContent
            sx={{
              display: 'flex',
              justifyContent: 'center',
              alignItems: 'center',
              flexDirection: 'column',
            }}
          >
            <Typography
              variant='h2'
              component='div'
              sx={{ fontWeight: 'bold' }}
            >
              Login to FYP
            </Typography>
            <LoginForm />
          </CardContent>
          <CardActions
            sx={{
              display: 'flex',
              justifyContent: 'center',
              alignItems: 'center',
              flexDirection: 'column',
              mb: 1,
            }}
          >
            <Button variant='outlined' size='large'>
              Login
            </Button>
            <Typography sx={{ mt: 1 }} variant='body2'>
              Not registered? Login here
            </Typography>
          </CardActions>
        </Card>
      </Container>
    </section>
  );
};

export default login;

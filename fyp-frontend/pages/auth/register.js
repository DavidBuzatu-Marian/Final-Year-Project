import React from 'react';
import Card from '@mui/material/Card';
import CardActions from '@mui/material/CardActions';
import CardContent from '@mui/material/CardContent';
import Button from '@mui/material/Button';
import Typography from '@mui/material/Typography';
import Container from '@mui/material/Container';
import TextField from '@mui/material/TextField';
import RegisterForm from '../../components/auth/registerForm';
import Link from 'next/link';
import style from '../../styles/Utils.module.scss';
const login = () => {
  return (
    <Container
      maxWidth='sm'
      sx={{
        minHeight: '100vh',
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
      }}
    >
      <header>
        <Typography variant='h3' sx={{ fontWeight: 'bold' }}>
          Sign up to project
        </Typography>
      </header>
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
              <RegisterForm />
            </CardContent>
            <CardActions
              sx={{
                mb: 1,
                ml: 1.25,
              }}
            >
              <Typography sx={{ mt: 1 }} variant='body2'>
                Already registered? Sign in{' '}
                <Link href='/auth/login'>
                  <a className={style.form_link}>here</a>
                </Link>
              </Typography>
            </CardActions>
          </Card>
        </Container>
      </section>
    </Container>
  );
};

export default login;

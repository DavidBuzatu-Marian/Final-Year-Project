import React, { useEffect } from 'react';
import Card from '@mui/material/Card';
import CardActions from '@mui/material/CardActions';
import CardContent from '@mui/material/CardContent';
import Button from '@mui/material/Button';
import Typography from '@mui/material/Typography';
import Container from '@mui/material/Container';
import TextField from '@mui/material/TextField';
import LoginForm from '../../components/auth/loginForm';
import Link from 'next/link';
import style from '../../styles/Utils.module.scss';
import { useUser } from '../../hooks/user';
import Router from 'next/router';
import { CircularProgress } from '@mui/material';
const Login = () => {
  const [user, { loading }] = useUser();
  useEffect(() => {
    if (user) {
      Router.push('/dashboard');
    }
  }, [user, loading]);

  return (
    <Container
      maxWidth='md'
      sx={{
        minHeight: '100vh',
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
      }}
    >
      {loading ? (
        <CircularProgress />
      ) : (
        <>
          <header>
            <Typography variant='h3' sx={{ fontWeight: 'bold' }}>
              Sign in to project
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
                  <LoginForm />
                </CardContent>
                <CardActions
                  sx={{
                    mb: 1,
                    ml: 1.25,
                  }}
                >
                  <Typography sx={{ mt: 1 }} variant='body2'>
                    Not registered? Sign up{' '}
                    <Link href='/auth/register'>
                      <a className={style.form_link}>here</a>
                    </Link>
                  </Typography>
                </CardActions>
              </Card>
            </Container>
          </section>
        </>
      )}
    </Container>
  );
};

export default Login;

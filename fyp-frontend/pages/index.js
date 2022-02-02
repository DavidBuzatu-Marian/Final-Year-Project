import Head from 'next/head';
import Button from '@mui/material/Button';
import Typography from '@mui/material/Typography';
import Container from '@mui/material/Container';
import { Grid } from '@mui/material';

export default function Home() {
  return (
    <Container
      sx={{
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
        minHeight: '100vh',
        textAlign: 'center',
      }}
    >
      <Head>
        <title>David-Marian Buzatu - Final Year Project</title>
        <meta
          name='description'
          content='Final Year Project of David-Marian Buzatu. Please sign in or sign up to access the features of the platform'
        />
        <link rel='icon' href='/favicon.ico' />
      </Head>

      <main>
        <Typography variant='h2' sx={{ fontWeight: 'bold' }}>
          Welcome to my dissertation. To access the website's features
        </Typography>

        <Container sx={{ mt: 4 }}>
          <Button
            href='/auth/register'
            variant='outlined'
            size='large'
            sx={{ mr: 2 }}
          >
            Sign Up
          </Button>
          <Button href='/auth/login' variant='contained' size='large'>
            Sign In
          </Button>
        </Container>
      </main>
    </Container>
  );
}

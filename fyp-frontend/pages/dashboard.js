import React, { useEffect } from 'react';
import Card from '@mui/material/Card';
import CardActions from '@mui/material/CardActions';
import CardContent from '@mui/material/CardContent';
import Button from '@mui/material/Button';
import Typography from '@mui/material/Typography';
import Container from '@mui/material/Container';
import TextField from '@mui/material/TextField';
import Link from 'next/link';
import style from '../styles/Utils.module.scss';
import { useUser } from '../hooks/user';
import Router from 'next/router';
const Dashboard = () => {
  const [user] = useUser();
  useEffect(() => {
    console.log(user);
    if (!user) {
      Router.push('/auth/login');
    }
  }, [user]);
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
      <header>
        <Typography variant='h3' sx={{ fontWeight: 'bold' }}>
          Dashboard
        </Typography>
      </header>
    </Container>
  );
};

export default Dashboard;

import React, { useEffect } from 'react';
import Typography from '@mui/material/Typography';
import Container from '@mui/material/Container';
import { useUser } from '../hooks/user';
import Router from 'next/router';
import DrawerMenu from '../components/dashboard/drawerMenu';
import CircularProgress from '@mui/material/CircularProgress';
import Box from '@mui/material/Box';

const Dashboard = () => {
  const [user, { loading }] = useUser();
  useEffect(() => {
    console.log(user);
    if (!user) {
      Router.push('/auth/login?afterLoginRedirect=/dashboard');
    }
  }, [user]);
  return (
    <Container
      maxWidth='100%'
      sx={{
        minHeight: '100vh',
        display: 'flex',
        mt: '4rem',
      }}
    >
      {loading ? (
        <CircularProgress />
      ) : (
        <>
          <DrawerMenu user={user} />
          <section>
            <Box
              component='main'
              sx={{ flexGrow: 1, bgcolor: 'background.default', p: 3 }}
            >
              <Typography variant='h3'>Dashboard</Typography>
            </Box>
          </section>
        </>
      )}
    </Container>
  );
};

export default Dashboard;

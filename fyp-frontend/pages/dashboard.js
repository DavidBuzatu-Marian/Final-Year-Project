import React, { useEffect } from 'react';
import Container from '@mui/material/Container';
import { useUser } from '../hooks/user';
import Router from 'next/router';
import DrawerMenu from '../components/dashboard/drawerMenu';
import CircularProgress from '@mui/material/CircularProgress';

import EnvironmentsDataGridHeader from '../components/dashboard/environmentsDataGridHeader';
import { Typography } from '@mui/material';
const Dashboard = () => {
  const [user, { loading }] = useUser();
  useEffect(() => {
    if (!user || user === null) {
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
      ) : user ? (
        <>
          <DrawerMenu user={user} />
          <section style={{ width: '100%' }}>
            <EnvironmentsDataGridHeader />
          </section>
        </>
      ) : (
        <Typography variant='h3'>User not found</Typography>
      )}
    </Container>
  );
};

export default Dashboard;

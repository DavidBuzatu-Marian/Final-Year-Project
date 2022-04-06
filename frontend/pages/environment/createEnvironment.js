import React, { useEffect, useState } from 'react';
import Container from '@mui/material/Container';
import { useUser } from '../../hooks/user';
import Router from 'next/router';
import DrawerMenu from '../../components/dashboard/drawerMenu';
import CircularProgress from '@mui/material/CircularProgress';
import CreateEnvironmentForm from '../../components/instance/createEnvironmentForm';
import EnvironmentHeader from '../../components/instance/environmentHeader';

const CreateEnvironment = () => {
  const [user, { loading }] = useUser();

  useEffect(() => {
    if (!user || user === null) {
      Router.push(
        '/auth/login?afterLoginRedirect=/environment/createEnvironment'
      );
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
      {loading || !user ? (
        <CircularProgress />
      ) : (
        <>
          <DrawerMenu user={user} />
          <section style={{ width: '100%' }}>
            <EnvironmentHeader headerTitle={'Create environment'} />
            <CreateEnvironmentForm />
          </section>
        </>
      )}
    </Container>
  );
};

export default CreateEnvironment;

import React, { useEffect, useState } from 'react';
import Container from '@mui/material/Container';
import { useUser } from '../hooks/user';
import Router from 'next/router';
import DrawerMenu from '../components/dashboard/drawerMenu';
import CircularProgress from '@mui/material/CircularProgress';
import DatasetsDataGrid from '../components/dataset/datasetsDataGrid';
import DatasetsDataGridHeader from '../components/dataset/datasetsDataGridHeader';

const Datasets = () => {
  const [user, { loading }] = useUser();
  const [selectedRow, setSelectedRow] = useState({});

  useEffect(() => {
    if (!user || user === null) {
      Router.push('/auth/login?afterLoginRedirect=/datasets');
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
            <DatasetsDataGridHeader selectedRow={selectedRow} />
            <DatasetsDataGrid setSelectedRow={setSelectedRow} />
          </section>
        </>
      )}
    </Container>
  );
};

export default Datasets;

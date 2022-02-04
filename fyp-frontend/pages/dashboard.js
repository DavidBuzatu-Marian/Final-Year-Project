import React, { useEffect } from 'react';
import Typography from '@mui/material/Typography';
import Container from '@mui/material/Container';
import { useUser } from '../hooks/user';
import Router from 'next/router';

import Box from '@mui/material/Box';
import Drawer from '@mui/material/Drawer';
import AppBar from '@mui/material/AppBar';
import Toolbar from '@mui/material/Toolbar';
import List from '@mui/material/List';
import Divider from '@mui/material/Divider';
import ListItem from '@mui/material/ListItem';
import ListItemIcon from '@mui/material/ListItemIcon';
import ListItemText from '@mui/material/ListItemText';
import CircularProgress from '@mui/material/CircularProgress';

const drawerWidth = 240;

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
          <header>
            <Box sx={{ display: 'flex' }}>
              <AppBar
                position='fixed'
                sx={{
                  width: `calc(100% - ${drawerWidth}px)`,
                  ml: `${drawerWidth}px`,
                }}
              >
                <Toolbar sx={{ justifyContent: 'end' }}>
                  <Typography variant='h6' noWrap component='div'>
                    {user.email}
                  </Typography>
                </Toolbar>
              </AppBar>
              <Drawer
                sx={{
                  width: drawerWidth,
                  flexShrink: 0,
                  '& .MuiDrawer-paper': {
                    width: drawerWidth,
                    boxSizing: 'border-box',
                  },
                }}
                variant='permanent'
                anchor='left'
              >
                <Toolbar>
                  <Typography variant='h6' noWrap component='div'>
                    Final Year Project
                  </Typography>
                </Toolbar>
                <Divider />
                <List>
                  {[
                    ['Environments', 'grid_view'],
                    ['Datasets', 'storage'],
                  ].map((item, index) => (
                    <ListItem button key={index}>
                      <ListItemIcon>
                        <span className='material-icons'>{item[1]}</span>
                      </ListItemIcon>
                      <ListItemText primary={item[0]} />
                    </ListItem>
                  ))}
                </List>
                <Divider />
                <List>
                  {[['Logout', 'logout']].map((item, index) => (
                    <ListItem button key={index}>
                      <ListItemIcon>
                        <span className='material-icons'>{item[1]}</span>
                      </ListItemIcon>
                      <ListItemText primary={item[0]} />
                    </ListItem>
                  ))}
                </List>
              </Drawer>
            </Box>
          </header>
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

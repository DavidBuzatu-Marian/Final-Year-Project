import React from 'react';
import Box from '@mui/material/Box';
import Drawer from '@mui/material/Drawer';
import AppBar from '@mui/material/AppBar';
import Toolbar from '@mui/material/Toolbar';
import List from '@mui/material/List';
import Divider from '@mui/material/Divider';
import ListItemButton from '@mui/material/ListItemButton';
import ListItemIcon from '@mui/material/ListItemIcon';
import ListItem from '@mui/material/ListItem';
import ListItemText from '@mui/material/ListItemText';
import Typography from '@mui/material/Typography';
import { logout } from '../auth/hooks';
import { useUser } from '../../hooks/user';
import Link from 'next/link';
import { useRouter } from 'next/router';

const drawerWidth = 240;

const DrawerMenu = ({ user }) => {
  const [_, { mutate }] = useUser();
  const router = useRouter();
  const [selectedIndex, setSelectedIndex] = React.useState(
    router.pathname.includes('/dashboard') ||
      router.pathname.includes('/environment')
      ? 0
      : 1
  );
  // As guided from documentation: https://mui.com/components/lists/
  const onClick = (event, index) => {
    setSelectedIndex(index);
  };

  const logoutUser = async () => {
    await logout();
    mutate(null);
  };

  return (
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
            <Typography variant='p' noWrap component='div'>
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
              ['Environments', 'grid_view', '/dashboard'],
              ['Datasets', 'storage', '/datasets'],
            ].map((item, index) => (
              <Link key={index} href={`${item[2]}`}>
                <ListItemButton
                  selected={selectedIndex === index}
                  onClick={(event) => onClick(event, index)}
                >
                  <ListItemIcon>
                    <span className='material-icons'>{item[1]}</span>
                  </ListItemIcon>
                  <ListItemText primary={item[0]} />
                </ListItemButton>
              </Link>
            ))}
          </List>
          <Divider />
          <List>
            {[['Logout', 'logout']].map((item, index) => (
              <ListItemButton key={index} onClick={(event) => logoutUser()}>
                <ListItemIcon>
                  <span className='material-icons'>{item[1]}</span>
                </ListItemIcon>
                <ListItemText primary={item[0]} />
              </ListItemButton>
            ))}
          </List>
        </Drawer>
      </Box>
    </header>
  );
};

export default DrawerMenu;

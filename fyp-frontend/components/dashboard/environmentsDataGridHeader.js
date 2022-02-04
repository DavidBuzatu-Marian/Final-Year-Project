import React from 'react';
import Box from '@mui/material/Box';
import { Divider } from '@mui/material';
import Toolbar from '@mui/material/Toolbar';
import EnvironmentsDataGrid from './environmentsDataGrid';
import Stack from '@mui/material/Stack';
import Button from '@mui/material/Button';
import Typography from '@mui/material/Typography';

const EnvironmentsDataGridHeader = () => {
  return (
    <Box
      component='main'
      sx={{
        bgcolor: 'background.default',
        p: 3,
      }}
    >
      <Toolbar sx={{ justifyContent: 'start' }}>
        <Typography variant='h6' noWrap component='div'>
          Environments
        </Typography>
        <Stack direction='row' spacing={2} sx={{ ml: 4 }}>
          <Button
            variant='outlined'
            startIcon={<span class='material-icons'>add</span>}
          >
            Create environment
          </Button>
          <Button
            variant='contained'
            startIcon={<span class='material-icons'>delete</span>}
          >
            Delete environment
          </Button>
        </Stack>
      </Toolbar>
      <Divider />
      <EnvironmentsDataGrid
        data={[{ id: 1, lastName: 'Snow', firstName: 'Jon', age: 35 }]}
      />
    </Box>
  );
};

export default EnvironmentsDataGridHeader;

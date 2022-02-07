import React from 'react';
import Box from '@mui/material/Box';
import { Divider } from '@mui/material';
import Toolbar from '@mui/material/Toolbar';
import Stack from '@mui/material/Stack';
import Button from '@mui/material/Button';
import Typography from '@mui/material/Typography';

const DatasetsDataGridHeader = () => {
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
          Data distribution
        </Typography>
        <Stack direction='row' spacing={2} sx={{ ml: 4 }}>
          <Button
            variant='outlined'
            startIcon={<span className='material-icons'>add</span>}
          >
            Add data
          </Button>
          <Button
            variant='contained'
            startIcon={<span className='material-icons'>add</span>}
          >
            Add training distribution
          </Button>
        </Stack>
      </Toolbar>
      <Divider />
    </Box>
  );
};

export default DatasetsDataGridHeader;

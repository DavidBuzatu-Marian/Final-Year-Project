import {
  Box,
  Button,
  Divider,
  FormControl,
  TextField,
  Typography,
} from '@mui/material';

import React from 'react';
import EnvironmentOptions from './environmentOptions';
import EnvironmentSelectionTabs from './environmentSelectionTabs';

const CreateEnvironmentForm = () => {
  return (
    <Box
      component='form'
      sx={{
        '& .MuiTextField-root': { width: '35ch', my: 1 },
        mt: 1,
        ml: 3,
      }}
    >
      <FormControl>
        <Typography variant='h5'>Machines configuration</Typography>
        <TextField
          id='outlined-required'
          label='Number of instances'
          type={'number'}
          sx={{ mt: '1rem !important' }}
        />
        <Divider />
        <EnvironmentSelectionTabs />
        <Divider />
        <EnvironmentOptions />
        <Divider />
        <Button variant='outlined' sx={{ mt: '1rem' }}>
          Create
        </Button>
        <Button
          variant='contained'
          sx={{ mt: '1rem' }}
          color='error'
          href='/dashboard'
        >
          Cancel
        </Button>
      </FormControl>
    </Box>
  );
};

export default CreateEnvironmentForm;

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
import axios from 'axios';
import { getConfig } from '../../config/defaultConfig';
import Link from 'next/link';

const CreateEnvironmentForm = () => {
  const [formValues, setFormValues] = React.useState({
    nr_instances: 1,
    environment_options: [],
    machine_series: 'e2',
    machine_type: 'e2-micro',
  });

  const onSubmit = async (event) => {
    // event.preventDefault();
    try {
      const res = await axios.post(
        getConfig()['environmentCreateUrl'],
        {
          ...formValues,
        },
        { withCredentials: true }
      );
    } catch (error) {
      console.log(error);
    }
  };

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
          value={formValues.nr_instances}
          onChange={(event) =>
            setFormValues({ ...formValues, nr_instances: event.target.value })
          }
          sx={{ mt: '1rem !important' }}
        />
        <Divider />
        <EnvironmentSelectionTabs />
        <Divider />
        <EnvironmentOptions
          setParentFormValues={setFormValues}
          parentFormValues={formValues}
          nrInstances={formValues.nr_instances}
        />
        <Divider />
        <Link href='/dashboard'>
          <Button
            variant='outlined'
            sx={{ mt: '1rem' }}
            onClick={(event) => onSubmit(event)}
          >
            Create
          </Button>
        </Link>
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

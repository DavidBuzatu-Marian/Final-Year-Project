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
  const [formValues, setFormValues] = React.useState({
    nr_instances: 1,
    environment_options: [],
    machine_series: 'e2',
    machine_type: 'e2-micro',
  });

  const onSubmit = () => {
    console.log(formValues);
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
        <Button
          variant='outlined'
          sx={{ mt: '1rem' }}
          onClick={(event) => onSubmit()}
        >
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

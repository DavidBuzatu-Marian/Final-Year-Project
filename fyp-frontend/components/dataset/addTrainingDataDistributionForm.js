import { FormControl, Stack, TextField } from '@mui/material';
import { Box } from '@mui/system';
import React from 'react';

const AddTrainingDataDistributionForm = ({ formValues, setFormValues }) => {
  const handleChange = (field, subField, idx) => (event) => {
    if (subField) {
      const newProp = formValues[field];
      newProp[idx][subField] = parseInt(event.target.value);
      setFormValues({
        ...formValues,
        newProp,
      });
    } else {
      setFormValues({
        ...formValues,
        [field]: event.target.value,
      });
    }
  };

  return (
    <Box
      component='form'
      sx={{
        '& .MuiTextField-root': { width: '35ch', my: 1 },
        mt: 1,
        mx: 'auto',
      }}
      autoComplete='off'
    >
      <FormControl>
        {formValues.data_distribution.map((instance, idx) => (
          <TextField
            id='outlined-required'
            key={Object.keys(instance)[0]}
            label={Object.keys(instance)[0]}
            type='number'
            value={formValues.data_distribution[idx][Object.keys(instance)[0]]}
            onChange={handleChange(
              'data_distribution',
              Object.keys(instance)[0],
              idx
            )}
          />
        ))}
      </FormControl>
    </Box>
  );
};

export default AddTrainingDataDistributionForm;

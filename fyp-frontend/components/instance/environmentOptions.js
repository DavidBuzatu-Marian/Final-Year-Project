import { Typography, TextField, Divider, Button } from '@mui/material';
import { Box } from '@mui/system';
import React from 'react';

const EnvironmentOptions = () => {
  //TODO: Get max nr of instances as prop

  const [formFields, setFormFields] = React.useState([
    [
      {
        label: 'Instance number',
        type: 'number',
        name: 'instanceNumber',
        min: 1,
        max: 10,
      },
      {
        label: 'Probability of failure',
        type: 'number',
        name: 'probabilityOfFailure',
        min: 1,
        max: 100,
      },
    ],
  ]);

  const [formValues, setFormValues] = React.useState([
    { instanceNumber: '', probabilityOfFailure: '' },
  ]);
  const addFormFields = () => {
    setFormFields([
      ...formFields,
      [
        {
          label: 'Instance number',
          type: 'number',
          name: 'instanceNumber',
          min: 1,
          max: 10,
        },
        {
          label: 'Probability of failure',
          type: 'number',
          name: 'probabilityOfFailure',
          min: 1,
          max: 100,
        },
      ],
    ]);
    setFormValues([
      ...formValues,
      { instanceNumber: '', probabilityOfFailure: '' },
    ]);
  };

  const onChange = (event, index) => {
    const formValuesCopy = [...formValues];
    formValuesCopy[index][event.target.name] = event.target.value;
    setFormValues(formValuesCopy);
  };

  const removeFormFields = (index) => {
    const formValuesCopy = [...formValues];
    formValuesCopy.splice(index, 1);
    setFormValues(formValuesCopy);
    const formFieldsCopy = [...formFields];
    formFieldsCopy.splice(index, 1);
    setFormFields(formFieldsCopy);
  };

  return (
    <Box sx={{ width: '100%', typography: 'body1', mt: '1rem' }}>
      <Typography variant='h6'>Environment options</Typography>
      {formFields.map((fields, idx) => {
        return (
          <div style={{ alignItems: 'center', display: 'flex' }}>
            {fields.map((field, id) => {
              return (
                <TextField
                  id='outlined-required'
                  key={idx + ',' + id}
                  label={field.label}
                  type={field.type}
                  name={field.name}
                  value={formValues[idx][field.name]}
                  onChange={(event) => onChange(event, idx)}
                  sx={{ mt: '1rem !important', mr: 1 }}
                  inputProps={{ min: field.min, max: field.max }}
                  //   onChange={(e) => {
                  //     const value = parseInt(e.target.value, 10);
                  //     if (value < min) {
                  //       // set value to min
                  //     } else if (value > max) {
                  //       // set value to max
                  //     }
                  //     // update state
                  //   }}
                />
              );
            })}
            <Button
              variant='outlined'
              onClick={(event) => removeFormFields(idx)}
              startIcon={<span className='material-icons'>delete</span>}
            >
              Remove
            </Button>
          </div>
        );
      })}
      <Button onClick={() => addFormFields()}>Add options</Button>
    </Box>
  );
};

export default EnvironmentOptions;

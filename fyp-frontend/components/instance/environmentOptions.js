import { Typography, TextField, Divider, Button } from '@mui/material';
import { Box } from '@mui/system';
import React from 'react';

const EnvironmentOptions = ({
  setParentFormValues,
  parentFormValues,
  nrInstances,
}) => {
  const [formFields, setFormFields] = React.useState([
    [
      {
        label: 'Instance number',
        type: 'number',
        name: 'instanceNumber',
        min: 1,
        max: 0,
      },
      {
        label: 'Probability of failure',
        type: 'number',
        name: 'probabilityOfFailure',
        min: 1,
        max: 100,
        endAdornment: <span className='material-icons'>percent</span>,
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
          max: 0,
        },
        {
          label: 'Probability of failure',
          type: 'number',
          name: 'probabilityOfFailure',
          min: 1,
          max: 100,
          endAdornment: <span className='material-icons'>percent</span>,
        },
      ],
    ]);
    setFormValues([
      ...formValues,
      { instanceNumber: '', probabilityOfFailure: '' },
    ]);
  };

  const onChange = (event, index, min, max) => {
    if (event.target.value.length === 0) {
      updateFormValues(event.target.name, index, '');
      return;
    }
    const value = parseInt(event.target.value, 10);
    if (value < min) {
      value = min;
    } else if (value > max) {
      value = max;
    }
    updateFormValues(event.target.name, index, value);
  };

  const updateFormValues = (name, index, value) => {
    const formValuesCopy = [...formValues];
    formValuesCopy[index][name] = value;
    setFormValues(formValuesCopy);
    setParentFormValues({
      ...parentFormValues,
      environment_options: formValuesCopy,
    });
  };

  const removeFormFields = (index) => {
    const formValuesCopy = [...formValues];
    formValuesCopy.splice(index, 1);
    setFormValues(formValuesCopy);
    setParentFormValues({
      ...parentFormValues,
      environment_options: formValuesCopy,
    });
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
                  onChange={(event) =>
                    onChange(
                      event,
                      idx,
                      field.min,
                      field.max === 0 ? nrInstances : field.max
                    )
                  }
                  sx={{ mt: '1rem !important', mr: 1 }}
                  InputProps={{
                    endAdornment: field.endAdornment && field.endAdornment,
                  }}
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

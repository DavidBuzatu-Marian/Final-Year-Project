import * as React from 'react';
import Box from '@mui/material/Box';
import TextField from '@mui/material/TextField';
import MenuItem from '@mui/material/MenuItem';

const GoogleMachineTypeSelector = ({ machineTypes }) => {
  const [machineType, setMachineType] = React.useState(machineTypes[0].value);

  const handleChange = (event) => {
    setMachineType(event.target.value);
  };

  return (
    <Box
      sx={{
        '& .MuiTextField-root': { mt: 1 },
      }}
      noValidate
      autoComplete='off'
    >
      <div>
        <TextField
          id='outlined-select-machine-type'
          select
          label='Machines type'
          value={machineType}
          onChange={handleChange}
          helperText='Please select your machine type'
        >
          {machineTypes.map((option) => (
            <MenuItem key={option.value} value={option.value}>
              {option.label}
            </MenuItem>
          ))}
        </TextField>
      </div>
    </Box>
  );
};
export default GoogleMachineTypeSelector;

import * as React from 'react';
import Box from '@mui/material/Box';
import TextField from '@mui/material/TextField';
import MenuItem from '@mui/material/MenuItem';

const GoogleMachineTypeSelector = ({
  machineTypesObject,
  machineSeriesList,
  formValues,
  setFormValues,
}) => {
  const [machineSeries, setMachineSeries] = React.useState(
    machineSeriesList[0].value
  );
  const [machineType, setMachineType] = React.useState(
    machineTypesObject[machineSeries][0].value
  );
  const [machineTypesList, setMachineTypesList] = React.useState(
    machineTypesObject[machineSeries]
  );

  const handleMachineTypeChange = (event) => {
    setMachineType(event.target.value);
  };

  const handleMachineSeriesChange = (event) => {
    setMachineSeries(event.target.value);
    setMachineType(machineTypesObject[event.target.value][0].value);
    setMachineTypesList(machineTypesObject[event.target.value]);
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
          id='outlined-select-machine-series'
          select
          label='Machines series'
          value={machineSeries}
          onChange={handleMachineSeriesChange}
          helperText='Please select your machine series'
        >
          {machineSeriesList.map((option) => (
            <MenuItem key={option.value} value={option.value}>
              {option.label}
            </MenuItem>
          ))}
        </TextField>
      </div>
      <div>
        <TextField
          id='outlined-select-machine-type'
          select
          label='Machines type'
          value={machineType}
          onChange={handleMachineTypeChange}
          helperText='Please select your machine type'
        >
          {machineTypesList.map((option) => (
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

import React from 'react';
import { TabContext, TabList, TabPanel } from '@mui/lab';
import { Tab, Box } from '@mui/material';
import GoogleMachineTypeSelector from './googleMachineTypeSelector';
import { machineTypesE2 } from '../utils/machineTypes';

const EnvironmentSelectionTabs = () => {
  const [value, setValue] = React.useState('1');

  const handleChange = (event, newValue) => {
    setValue(newValue);
  };
  return (
    <Box sx={{ width: '100%', typography: 'body1', mt: '1rem' }}>
      <TabContext value={value}>
        <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
          <TabList onChange={handleChange} aria-label='lab API tabs example'>
            <Tab label='General-Purpose' value='1' />
            <Tab label='Compute-Optimized' value='2' />
            <Tab label='Memory-Optimized' value='3' />
          </TabList>
        </Box>
        <TabPanel value='1' sx={{ p: 0, mt: 2 }}>
          <GoogleMachineTypeSelector machineTypes={machineTypesE2} />
        </TabPanel>
        <TabPanel value='2'>Item Two</TabPanel>
        <TabPanel value='3'>Item Three</TabPanel>
      </TabContext>
    </Box>
  );
};

export default EnvironmentSelectionTabs;

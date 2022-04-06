import React from 'react';
import { TabContext, TabList, TabPanel } from '@mui/lab';
import { Tab, Box } from '@mui/material';
import GoogleMachineTypeSelector from './googleMachineTypeSelector';
import {
  machineTypesC2,
  machineTypesE2,
  machineTypesN2,
} from '../utils/machineTypes';
import {
  computeOptimizedSeries,
  generalPurposeSeries,
} from '../utils/machineSeries';

const EnvironmentSelectionTabs = ({ formValues, setFormValues }) => {
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
          </TabList>
        </Box>
        <TabPanel value='1' sx={{ p: 0, mt: 2 }}>
          <GoogleMachineTypeSelector
            machineSeriesList={generalPurposeSeries}
            machineTypesObject={{ e2: machineTypesE2, n2: machineTypesN2 }}
            formValues={formValues}
            setFormValues={setFormValues}
          />
        </TabPanel>
        <TabPanel value='2' sx={{ p: 0, mt: 2 }}>
          <GoogleMachineTypeSelector
            machineSeriesList={computeOptimizedSeries}
            machineTypesObject={{ c2: machineTypesC2 }}
            formValues={formValues}
            setFormValues={setFormValues}
          />
        </TabPanel>
      </TabContext>
    </Box>
  );
};

export default EnvironmentSelectionTabs;

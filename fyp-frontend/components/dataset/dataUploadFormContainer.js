import React from "react";
import { Box, Tab } from "@mui/material";
import { TabContext, TabList, TabPanel } from "@mui/lab";
import AddDataForm from "./addDataForm";

const DataUploadFormContainer = ({ formValues, setFormValues }) => {
  const [value, setValue] = React.useState("1");

  const handleChange = (event, newValue) => {
    setValue(newValue);
  };

  return (
    <Box sx={{ width: "100%", typography: "body1", mt: "1rem" }}>
      <TabContext value={value}>
        <Box sx={{ borderBottom: 1, borderColor: "divider" }}>
          <TabList onChange={handleChange} aria-label="Type of data selection">
            <Tab label="Training dataset" value="1" />
            <Tab label="Validation dataset" value="2" />
            <Tab label="Testing dataset" value="3" />
          </TabList>
        </Box>
        <TabPanel value="1" sx={{ p: 0, mt: 2 }}>
          <AddDataForm formValues={formValues} setFormValues={setFormValues} />
        </TabPanel>
        <TabPanel value="2" sx={{ p: 0, mt: 2 }}>
          <AddDataForm formValues={formValues} setFormValues={setFormValues} />
        </TabPanel>
        <TabPanel value="3" sx={{ p: 0, mt: 2 }}>
          <AddDataForm formValues={formValues} setFormValues={setFormValues} />
        </TabPanel>
      </TabContext>
    </Box>
  );
};

export default DataUploadFormContainer;

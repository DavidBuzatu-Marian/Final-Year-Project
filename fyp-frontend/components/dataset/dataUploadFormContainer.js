import React from "react";
import { Box, Tab } from "@mui/material";
import { TabContext, TabList, TabPanel } from "@mui/lab";
import AddDataForm from "./addDataForm";

const dataNameMap = {
  1: {
    dataName: "train_data",
    labelsName: "train_labels",
    url: "environmentDataDistributionTrainAddUrl",
  },
  2: {
    dataName: "validation_data",
    labelsName: "validation_labels",
    url: "environmentDataDistributionValidationAddUrl",
  },
  3: {
    dataName: "test_data",
    labelsName: "test_labels",
    url: "environmentDataDistributionTestAddUrl",
  },
};

const DataUploadFormContainer = ({
  formValues,
  setFormValues,
  headerModals,
  setHeaderModalsState,
  activeHeaderModal,
}) => {
  const [value, setValue] = React.useState("1");

  const handleChange = (event, newValue) => {
    setValue(newValue);
    setFormValues({ ...formValues, ...dataNameMap[newValue] });
    setHeaderModalsState({
      ...headerModals,
      [activeHeaderModal]: {
        ...headerModals[activeHeaderModal],
        url: dataNameMap[newValue].url,
      },
    });
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

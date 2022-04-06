import {
  Box,
  Button,
  Divider,
  FormControl,
  TextField,
  Typography,
} from "@mui/material";

import React from "react";
import EnvironmentOptions from "./environmentOptions";
import EnvironmentSelectionTabs from "./environmentSelectionTabs";
import axios from "axios";
import { getConfig } from "../../config/defaultConfig";
import Link from "next/link";
import ModalProgress from "../utils/modalProgress";
import ClosableAlert from "../alert/closableAlert";

const CreateEnvironmentForm = () => {
  const [formValues, setFormValues] = React.useState({
    nr_instances: 1,
    environment_options: [],
    machine_series: "e2",
    machine_type: "e2-micro",
  });
  const [progressModal, setProgressModal] = React.useState({
    isVisible: false,
    jobLink: null,
  });

  const onSubmit = async (event) => {
    try {
      setProgressModal({ isVisible: true });
      const res = await axios.post(
        getConfig()["environmentCreateUrl"],
        {
          ...formValues,
        },
        { withCredentials: true }
      );
      setProgressModal({ isVisible: true, ...res.data });
    } catch (error) {
      setProgressModal({
        isVisible: false,
        errorMessage: "Something went wrong on our end. Please retry.",
      });
    }
  };

  return (
    <Box
      component="form"
      sx={{
        "& .MuiTextField-root": { width: "35ch", my: 1 },
        mt: 1,
        ml: 3,
      }}
    >
      {progressModal.errorMessage && (
        <ClosableAlert
          severity={"error"}
          alertMessage={progressModal.errorMessage}
        />
      )}
      <FormControl>
        <Typography variant="h5">Machines configuration</Typography>
        <TextField
          id="outlined-required"
          label="Number of instances"
          type={"number"}
          value={formValues.nr_instances}
          onChange={(event) =>
            setFormValues({ ...formValues, nr_instances: event.target.value })
          }
          sx={{ mt: "1rem !important" }}
        />
        <Divider />
        <EnvironmentSelectionTabs
          formValues={formValues}
          setFormValues={setFormValues}
        />
        <Divider />
        <EnvironmentOptions
          setParentFormValues={setFormValues}
          parentFormValues={formValues}
          nrInstances={formValues.nr_instances}
        />
        <Divider />
        <Button
          variant="outlined"
          sx={{ mt: "1rem" }}
          onClick={(event) => onSubmit(event)}
        >
          Create
        </Button>
        <Button
          variant="contained"
          sx={{ mt: "1rem" }}
          color="error"
          href="/dashboard"
        >
          Cancel
        </Button>
      </FormControl>
      <ModalProgress
        isOpen={progressModal.isVisible}
        modalButtonText={"Go to Dashboard"}
        modalTitle={"Environment creation progress"}
        modalContent={"Creating environment..."}
        modalAlertMessage={"Environment creation process started!"}
        redirectUrl={"/dashboard"}
        jobLink={progressModal.jobLink}
      />
    </Box>
  );
};

export default CreateEnvironmentForm;

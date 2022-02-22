import { FormControl, Stack, TextField } from "@mui/material";
import { Box } from "@mui/system";
import React from "react";

const AddTrainingDataDistributionForm = ({ formValues, setFormValues }) => {
  const handleChange = (field, subField) => (event) => {
    const newValue =
      event.target.value.length === 0 ? 0 : parseInt(event.target.value);
    if (subField) {
      const newProp = formValues[field];
      newProp[subField] = newValue;
      setFormValues({
        ...formValues,
        newProp,
      });
    } else {
      setFormValues({
        ...formValues,
        [field]: newValue,
      });
    }
  };

  return (
    <Box
      component="form"
      sx={{
        "& .MuiTextField-root": { width: "35ch", my: 1 },
        mt: 1,
        mx: "auto",
      }}
    >
      <FormControl>
        <TextField
          id="outlined-required"
          label="Dataset length"
          type="number"
          value={formValues.dataset_length}
          onChange={handleChange("dataset_length")}
        />
        {Object.keys(formValues.data_distribution).map((instance) => (
          <TextField
            id="outlined-required"
            key={instance}
            label={instance}
            type="number"
            value={formValues.data_distribution[instance]}
            onChange={handleChange("data_distribution", instance)}
          />
        ))}
      </FormControl>
    </Box>
  );
};

export default AddTrainingDataDistributionForm;

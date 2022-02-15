import { Button, FormControl, Stack, Typography, Fab } from "@mui/material";
import { Box } from "@mui/system";
import React from "react";

const AddDataForm = ({ formValues, setFormValues }) => {
  // hidden file input inspiration from: https://medium.com/web-dev-survey-from-kyoto/how-to-customize-the-file-upload-button-in-react-b3866a5973d8
  const hiddenDataFilesInput = React.useRef(null);
  const hiddenLabelFilesInput = React.useRef(null);

  const onClick = (hiddenFileInput) => {
    hiddenFileInput.current.click();
  };

  const handleChange = (event, name) => {
    const filesUploaded = event.target.files;
    console.log(filesUploaded);
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
      <FormControl sx={{ width: "100%" }}>
        <Stack
          direction="row"
          sx={{
            alignItems: "center",
            width: "100%",
            justifyContent: "center",
            mb: 2,
          }}
        >
          <Typography variant="p">Upload data image(s)</Typography>
          <Fab
            color="primary"
            aria-label="add"
            onClick={(event) => onClick(hiddenDataFilesInput)}
            sx={{ ml: 2 }}
          >
            <span className="material-icons">add</span>
          </Fab>
          <input
            type="file"
            ref={hiddenDataFilesInput}
            onChange={(event) => handleChange(event, formValues.dataName)}
            style={{ display: "none" }}
            multiple
          />
        </Stack>
        <Stack
          direction="row"
          sx={{ alignItems: "center", width: "100%", justifyContent: "center" }}
        >
          <Typography variant="p">Upload labels image(s)</Typography>
          <Fab
            color="primary"
            aria-label="add"
            onClick={(event) => onClick(hiddenLabelFilesInput)}
            sx={{ ml: 2 }}
          >
            <span className="material-icons">add</span>
          </Fab>
          <input
            type="file"
            ref={hiddenLabelFilesInput}
            onChange={(event) => handleChange(event, formValues.labelsName)}
            style={{ display: "none" }}
            multiple
          />
        </Stack>
      </FormControl>
    </Box>
  );
};

export default AddDataForm;

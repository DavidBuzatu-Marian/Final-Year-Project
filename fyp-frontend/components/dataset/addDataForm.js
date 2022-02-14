import { Button, FormControl, Stack, Typography, Fab } from '@mui/material';
import { Box } from '@mui/system';
import React from 'react';

const AddDataForm = ({ formValues, setFormValues }) => {
  // hidden file input inspiration from: https://medium.com/web-dev-survey-from-kyoto/how-to-customize-the-file-upload-button-in-react-b3866a5973d8
  const hiddenFileInput = React.useRef(null);

  const onClick = () => {
    hiddenFileInput.current.click();
  };

  const handleChange = (event) => {
    const filesUploaded = event.target.files;
    console.log(filesUploaded);
  };

  return (
    <Box
      component='form'
      sx={{
        '& .MuiTextField-root': { width: '35ch', my: 1 },
        mt: 1,
        mx: 'auto',
      }}
    >
      <FormControl>
        <Stack>
          <Typography variant='p'>Upload data image(s)</Typography>
          <Fab color='primary' aria-label='add' onClick={(event) => onClick()}>
            <span className='material-icons'>add</span>
          </Fab>
          <input
            type='file'
            ref={hiddenFileInput}
            onChange={handleChange(formValues.dataName)}
          />
        </Stack>
        <Stack>
          <Typography variant='p'>Upload labels image(s)</Typography>
          <Fab color='primary' aria-label='add' onClick={(event) => onClick()}>
            <span className='material-icons'>add</span>
          </Fab>
          <input
            type='file'
            ref={hiddenFileInput}
            onChange={handleChange(formValues.labelsName)}
          />
        </Stack>
      </FormControl>
    </Box>
  );
};

export default AddDataForm;

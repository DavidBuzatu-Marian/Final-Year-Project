import * as React from 'react';
import Box from '@mui/material/Box';
import TextField from '@mui/material/TextField';

export default function LoginForm() {
  return (
    <Box
      component='form'
      sx={{
        '& .MuiTextField-root': { width: '25ch', my: 1 },
        mt: 4,
        mx: 'auto',
        mb: 4,
      }}
      noValidate
      autoComplete='off'
    >
      <div>
        <TextField id='outlined-required' label='Email' />
      </div>
      <div>
        <TextField
          id='outlined-password-input'
          label='Password'
          type='password'
          autoComplete='current-password'
        />
      </div>
    </Box>
  );
}

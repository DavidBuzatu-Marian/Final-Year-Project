import * as React from 'react';
import Box from '@mui/material/Box';
import TextField from '@mui/material/TextField';
import Button from '@mui/material/Button';
import { handleClickShowPassword, handleMouseDownPassword } from './hooks';
import InputAdornment from '@mui/material/InputAdornment';
import IconButton from '@mui/material/IconButton';
import FormControl from '@mui/material/FormControl';
import LoadingButton from '@mui/lab/LoadingButton';
export default function LoginForm() {
  const [formValues, setFormValues] = React.useState({
    email: '',
    password: '',
    showPassword: false,
  });

  const handleChange = (prop) => (event) => {
    setFormValues({
      ...formValues,
      [prop]: event.target.value,
    });
  };

  const checkErrors = () => {
    return !(formValues.email.length > 0 && formValues.password.length > 0);
  };
  return (
    <Box
      component='form'
      sx={{
        '& .MuiTextField-root': { width: '35ch', my: 1 },
        mt: 1,
        mx: 'auto',
      }}
      noValidate
      autoComplete='off'
    >
      <FormControl>
        <TextField
          id='outlined-required'
          label='Email'
          onChange={handleChange('email')}
        />
        <TextField
          id='outlined-required'
          label='Password'
          type={formValues.showPassword ? 'text' : 'password'}
          value={formValues.password}
          onChange={handleChange('password')}
          InputProps={{
            endAdornment: (
              <InputAdornment position='end'>
                <IconButton
                  aria-label='toggle password visibility'
                  onClick={() =>
                    handleClickShowPassword(setFormValues, formValues)
                  }
                  onMouseDown={handleMouseDownPassword}
                  edge='end'
                >
                  {formValues.showPassword ? (
                    <span className='material-icons'>visibility</span>
                  ) : (
                    <span className='material-icons'>visibility_off</span>
                  )}
                </IconButton>
              </InputAdornment>
            ),
          }}
        />
      </FormControl>
      <div style={{ marginTop: '1rem' }}>
        <LoadingButton
          variant='outlined'
          size='large'
          disabled={checkErrors()}
          endIcon={
            <span className='material-icons' loadingPosition='end'>
              login
            </span>
          }
        >
          Login
        </LoadingButton>
      </div>
    </Box>
  );
}

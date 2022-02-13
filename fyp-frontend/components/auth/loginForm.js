import * as React from 'react';
import Box from '@mui/material/Box';
import TextField from '@mui/material/TextField';
import { handleClickShowPassword, handleMouseDownPassword } from './hooks';
import InputAdornment from '@mui/material/InputAdornment';
import IconButton from '@mui/material/IconButton';
import FormControl from '@mui/material/FormControl';
import LoadingButton from '@mui/lab/LoadingButton';
import { useUser } from '../../hooks/user';
import axios from 'axios';
import { getConfig } from '../../config/defaultConfig';
import ClosableAlert from '../alert/closableAlert';

export default function LoginForm() {
  const [formValues, setFormValues] = React.useState({
    email: '',
    password: '',
    showPassword: false,
    onSubmitError: '',
    alertId: 0,
    loading: false,
  });
  const [user, { mutate }, error] = useUser();

  const handleChange = (prop) => (event) => {
    setFormValues({
      ...formValues,
      [prop]: event.target.value,
    });
  };

  const checkErrors = () => {
    return !(formValues.email.length > 0 && formValues.password.length > 0);
  };

  const onSubmit = async (event) => {
    event.preventDefault();
    setFormValues({ ...formValues, loading: true });
    let onSubmitError = false;
    try {
      const res = await axios.post(
        getConfig()['loginUrl'],
        {
          username: formValues.email,
          email: formValues.email,
          password: formValues.password,
        },
        { withCredentials: true }
      );
      mutate(res.data);
    } catch (err) {
      onSubmitError = true;
    } finally {
      setFormValues({
        ...formValues,
        password: '',
        onSubmitError,
        loading: false,
        alertId: formValues.alertId + 1,
      });
    }
  };

  return (
    <Box
      component='form'
      sx={{
        '& .MuiTextField-root': { width: '35ch', my: 1 },
        mt: 1,
        mx: 'auto',
      }}
      autoComplete='off'
    >
      <FormControl>
        {formValues.onSubmitError && (
          <ClosableAlert
            key={formValues.id}
            severity={'error'}
            alertMessage={'Signing in went wrong. Invalid email or password'}
          />
        )}
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
        <LoadingButton
          variant='outlined'
          disabled={checkErrors()}
          loadingPosition='end'
          loading={formValues.loading}
          endIcon={<span className='material-icons'>login</span>}
          sx={{ mt: '1rem' }}
          onClick={(event) => onSubmit(event)}
        >
          Login
        </LoadingButton>
      </FormControl>
    </Box>
  );
}

import * as React from 'react';
import Box from '@mui/material/Box';
import TextField from '@mui/material/TextField';
import LoadingButton from '@mui/lab/LoadingButton';
import { FormControl } from '@mui/material';
import InputAdornment from '@mui/material/InputAdornment';
import IconButton from '@mui/material/IconButton';
import { handleClickShowPassword, handleMouseDownPassword } from './hooks';
import axios from 'axios';
import { getConfig } from '../../config/defaultConfig';
import ClosableAlert from '../alert/closableAlert';

export default function RegisterForm() {
  const [formValues, setFormValues] = React.useState({
    email: '',
    password: '',
    passwordConfirm: '',
    showPassword: false,
    passwordErrorText: '',
    passwordConfirmErrorText: '',
    emailErrorText: '',
    loading: false,
    onSubmitError: false,
  });

  /* Added as per described in official documentation: https://mui.com/components/text-fields/#form-props
   * Made slight adaptations
   */
  const handleChange = (prop) => (event) => {
    if (prop === 'password') {
      const err = validatePassword(event.target.value);
      setFormValues({
        ...formValues,
        [prop]: event.target.value,
        passwordErrorText: err,
      });
    } else if (prop === 'passwordConfirm') {
      const err =
        formValues.password !== event.target.value
          ? 'Passwords must match!'
          : '';
      setFormValues({
        ...formValues,
        [prop]: event.target.value,
        passwordConfirmErrorText: err,
      });
    } else {
      const err = validateEmail(event.target.value);
      setFormValues({
        ...formValues,
        [prop]: event.target.value,
        emailErrorText: err,
      });
    }
  };

  /* RegEx from: https://regexr.com/3bfsi*/
  const validatePassword = (password) => {
    const regEx = '^(?=.*[0-9])(?=.*[a-z])(?=.*[A-Z])(?=.*[a-zA-Z0-9]).{8,}$';
    const errorMessage =
      password.length > 0 && password.match(regEx) === null
        ? 'Password needs to be at least 8 characters long, contain one uppercase and lowercase letter, and a number'
        : '';
    return errorMessage;
  };

  /* regex from: https://stackoverflow.com/a/46181/11023871 */
  const validateEmail = (email) => {
    return String(email)
      .toLowerCase()
      .match(
        /^(([^<>()[\]\\.,;:\s@"]+(\.[^<>()[\]\\.,;:\s@"]+)*)|(".+"))@((\[[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\])|(([a-zA-Z\-0-9]+\.)+[a-zA-Z]{2,}))$/
      ) === null
      ? 'Email is not valid'
      : '';
  };

  const checkErrors = () => {
    return !(
      formValues.passwordErrorText.length === 0 &&
      formValues.passwordConfirmErrorText.length === 0 &&
      formValues.emailErrorText.length === 0 &&
      formValues.email.length > 0 &&
      formValues.password.length > 0 &&
      formValues.passwordConfirm.length > 0
    );
  };

  const onSubmit = async (event) => {
    event.preventDefault();

    setFormValues({ ...formValues, loading: true });
    let onSubmitError = false;
    try {
      const res = await axios.post(getConfig()['registerUrl'], {
        email: formValues.email,
        password: formValues.password,
      });
      console.log(res);
    } catch (err) {
      onSubmitError = true;
      console.log(err);
    } finally {
      setFormValues({
        ...formValues,
        password: '',
        passwordConfirm: '',
        loading: false,
        onSubmitError,
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
            severity={'error'}
            alertMessage={'Registration went wrong'}
          />
        )}

        <TextField
          id='outlined-required'
          label='Email'
          value={formValues.email}
          onChange={handleChange('email')}
          error={formValues.emailErrorText.length > 0}
          helperText={formValues.emailErrorText}
        />
        <TextField
          id='outlined-required'
          label='Password'
          type={formValues.showPassword ? 'text' : 'password'}
          value={formValues.password}
          onChange={handleChange('password')}
          error={formValues.passwordErrorText.length > 0}
          helperText={formValues.passwordErrorText}
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

        <TextField
          id='outlined-required'
          label='Password confirm'
          type={formValues.showPassword ? 'text' : 'password'}
          value={formValues.passwordConfirm}
          onChange={handleChange('passwordConfirm')}
          error={formValues.passwordConfirmErrorText.length > 0}
          helperText={formValues.passwordConfirmErrorText}
        />
        <LoadingButton
          variant='outlined'
          disabled={checkErrors()}
          loadingPosition='end'
          loading={formValues.loading}
          endIcon={<span className='material-icons'>send</span>}
          sx={{ marginTop: '1rem' }}
          onClick={(event) => onSubmit(event)}
        >
          Register
        </LoadingButton>
      </FormControl>
    </Box>
  );
}

import axios from 'axios';
import { getConfig } from '../../config/defaultConfig';

export const handleClickShowPassword = (setFormValues, formValues) => {
  setFormValues({
    ...formValues,
    showPassword: !formValues.showPassword,
  });
};

export const handleMouseDownPassword = (event) => {
  event.preventDefault();
};

export const logout = async () => {
  const res = await axios.post(
    getConfig()['logoutUrl'],
    {},
    { withCredentials: true }
  );
  return res;
};

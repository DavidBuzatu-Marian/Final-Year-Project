export const handleClickShowPassword = (setFormValues, formValues) => {
  setFormValues({
    ...formValues,
    showPassword: !formValues.showPassword,
  });
};

export const handleMouseDownPassword = (event) => {
  event.preventDefault();
};

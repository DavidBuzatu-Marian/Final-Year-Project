import { Box } from "@mui/material";
import React from "react";
import JSONInput from "react-json-editor-ajrm";
import locale from "react-json-editor-ajrm/locale/en";

const AddModelForm = () => {
  return (
    <Box
      component="form"
      sx={{
        mt: 1,
        mx: "auto",
      }}
    >
      <JSONInput
        id="a_unique_id"
        placeholder={[
          'Specify your options using the following structure: {layer: {layer_type: "", subtype: "", parameters: {...}}}',
        ]}
        theme="light_mitsuketa_tribute"
        locale={locale}
        height="550px"
      />
    </Box>
  );
};

export default AddModelForm;

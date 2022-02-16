import { Box, Typography } from "@mui/material";
import React from "react";
import JSONInput from "react-json-editor-ajrm";
import locale from "react-json-editor-ajrm/locale/en";

const AddModelForm = ({ formValues, setFormValues }) => {
  return (
    <Box
      component="form"
      sx={{
        mt: 1,
        mx: "auto",
      }}
    >
      <Typography variant="h6" sx={{ whiteSpace: "pre-wrap" }}>
        Specify your options using the following structure:
      </Typography>
      <JSONInput
        id="a_unique_id"
        placeholder={[
          {
            layer: {
              layer_type: "Convolution",
              subtype: "Conv2d",
              parameters: {
                in_channels: 1,
                out_channels: 4,
                kernel_size: 3,
                stride: 1,
                padding: 1,
              },
            },
          },
          {
            layer: {
              layer_type: "Convolution",
              subtype: "Conv2d",
              parameters: {
                in_channels: 1,
                out_channels: 4,
                kernel_size: 3,
                stride: 1,
                padding: 1,
              },
            },
          },
        ]}
        theme="light_mitsuketa_tribute"
        locale={locale}
        height="450px"
        onChange={(event) =>
          setFormValues({
            ...formValues,
            environment_model_network_options: { network: event.jsObject },
          })
        }
      />
    </Box>
  );
};

export default AddModelForm;

import { Box, Typography, Badge } from "@mui/material";
import React from "react";
import JSONInput from "react-json-editor-ajrm";
import locale from "react-json-editor-ajrm/locale/en";
import ModelLayersList from "./modelLayersList";

const AddModelForm = ({ formValues, setFormValues }) => {
  return (
    <Box
      component="form"
      sx={{
        mt: 1,
        mx: "auto",
      }}
    >
      <ModelLayersList
        listOptionsInit={{
          listOptions: [
            {
              name: "Convolution",
              values: [
                "Conv1d",
                "Conv2d",
                "Conv3d",
                "ConvTranspose1d",
                "ConvTranspose2d",
                "ConvTranspose3d",
                "Fold",
                "Unfold",
              ],
            },
            {
              name: "Pooling",
              values: [
                "MaxPool1d",
                "MaxPool2d",
                "MaxPool3d",
                "MaxUnpool1d",
                "MaxUnpool2d",
                "MaxUnpool3d",
                "AvgPool1d",
                "AvgPool2d",
                "AvgPool3d",
                "FractionalMaxPool2d",
                "FractionalMaxPool3d",
                "LPPool1d",
                "LPPool2d",
                "AdaptiveMaxPool1d",
                "AdaptiveMaxPool2d",
                "AdaptiveMaxPool3d",
                "AdaptiveAvgPool1d",
                "AdaptiveAvgPool2d",
                "AdaptiveAvgPool3d",
              ],
            },
            {
              name: "Linear",
              values: ["Identity", "Linear", "Bilinear", "LazyLinear"],
            },
          ],
          collapsables: {
            Convolution: false,
            Pooling: false,
            Linear: false,
          },
        }}
        title={"Layer types and the available options"}
      />
      <Typography variant="h6" sx={{ whiteSpace: "pre-wrap", mb: 1 }}>
        Specify your options using the following structure{" "}
        <a href="https://pytorch.org/docs/stable/nn.html" target="_blank">
          <Badge color="secondary" variant="dot" sx={{ cursor: "pointer" }}>
            <span className="material-icons" style={{ color: "#002884" }}>
              help
            </span>
          </Badge>
        </a>{" "}
        :
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

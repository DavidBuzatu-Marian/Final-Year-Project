import { Box, Typography, Badge, TextField } from "@mui/material";
import React from "react";
import JSONInput from "react-json-editor-ajrm";
import locale from "react-json-editor-ajrm/locale/en";
import OptionsDynamicList from "./optionsDynamicList";

const TrainModelForm = ({ formValues, setFormValues }) => {
  return (
    <Box
      component="form"
      sx={{
        mt: 1,
        mx: "auto",
      }}
    >
      <TextField
        id="outlined-required"
        label="Training iterations"
        type={"number"}
        value={formValues.training_iterations}
        onChange={(event) =>
          setFormValues({
            ...formValues,
            training_iterations: event.target.value,
          })
        }
        sx={{ width: "100%" }}
      />
      <OptionsDynamicList
        listOptionsInit={{
          listOptions: [
            {
              name: "Loss",
              values: [
                "L1Loss",
                "MSELoss",
                "CrossEntropyLoss",
                "NLLLoss",
                "PoissonNLLLoss",
                "GaussianNLLLoss",
                "BCELoss",
                "BCEWithLogitsLoss",
                "SoftMarginLoss",
                "MultiLabelSoftMarginLoss",
              ],
            },
            {
              name: "Optimiser",
              values: ["SGD", "RMSprop", "ASGD", "Adamax", "Adam", "Adagrad"],
            },
            {
              name: "Hyperparameters",
              values: [
                "epochs",
                "num_workers",
                "batch_size",
                "shuffle",
                "drop_last",
                "normalizer",
                "normalizer_mean",
                "normalizer_std",
                "reshape",
              ],
            },
          ],
          collapsables: {
            Loss: false,
            Optimiser: false,
            Hyperparameters: false,
          },
        }}
        title={"Losses, optimisers and hyperparameter options"}
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
        placeholder={{
          loss: {
            loss_type: "CrossEntropyLoss",
            parameters: {},
          },
          optimizer: {
            optimizer_type: "RMSprop",
            parameters: {},
          },
          hyperparameters: {
            epochs: 5,
            batch_size: 5,
            reshape: "5, 1, 96, 96",
            normalizer: true,
            normalizer_mean: "0.5",
            normalizer_std: "0.5",
            drop_last: true,
          },
        }}
        theme="light_mitsuketa_tribute"
        locale={locale}
        height="450px"
        onChange={(event) =>
          setFormValues({
            ...formValues,
            environment_parameters: event.jsObject,
          })
        }
      />
    </Box>
  );
};

export default TrainModelForm;

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
            training_iterations:
              event.target.value.length === 0
                ? 0
                : parseInt(event.target.value),
          })
        }
        sx={{ width: "100%" }}
      />
      <TextField
        id="outlined-required"
        label="Maximum trials for device availability"
        type={"number"}
        value={formValues["training_options"].max_trials}
        onChange={(event) =>
          setFormValues({
            ...formValues,
            ["training_options"]: {
              ...formValues["training_options"],
              max_trials:
                event.target.value.length === 0
                  ? 0
                  : parseInt(event.target.value),
            },
          })
        }
        sx={{ width: "100%", mt: 4 }}
      />
      <TextField
        id="outlined-required"
        label="Minimum number of required devices"
        type={"number"}
        value={formValues["training_options"].required_instances}
        onChange={(event) =>
          setFormValues({
            ...formValues,
            ["training_options"]: {
              ...formValues["training_options"],
              required_instances:
                event.target.value.length === 0
                  ? 0
                  : parseInt(event.target.value),
            },
          })
        }
        sx={{ width: "100%", mt: 4 }}
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
                "normalize",
                "standardize",
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
      <Typography variant="p" sx={{ whiteSpace: "pre-wrap", mb: 1 }}>
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
            parameters: {
              lr: 0.001,
              weight_decay: 0.00000001,
              momentum: 0.9,
            },
          },
          hyperparameters: {
            epochs: 60,
            batch_size: 4,
            reshape: "4, 1, 96, 96",
            standardize: true,
          },
        }}
        theme="light_mitsuketa_tribute"
        locale={locale}
        height="450px"
        width="100%"
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

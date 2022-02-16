import React from "react";
import Box from "@mui/material/Box";
import { Divider } from "@mui/material";
import Toolbar from "@mui/material/Toolbar";
import Stack from "@mui/material/Stack";
import Button from "@mui/material/Button";
import Typography from "@mui/material/Typography";
import ModalDistributionForm from "./modalDistributionForm";
import AddTrainingDataDistributionForm from "./addTrainingDataDistributionForm";
import DataUploadFormContainer from "./dataUploadFormContainer";

const DatasetsDataGridHeader = ({ selectedRow, user }) => {
  const [modals, setModals] = React.useState({
    trainingDistribution: {
      isVisible: false,
      url: "environmentTrainingDistributionAddUrl",
    },
    dataDistribution: {
      isVisible: false,
      url: "environmentDataDistributionTrainAddUrl",
    },
  });

  return (
    <Box
      component="main"
      sx={{
        bgcolor: "background.default",
        p: 3,
      }}
    >
      <Toolbar sx={{ justifyContent: "start" }}>
        <Typography variant="h6" noWrap component="div">
          Data distribution
        </Typography>
        <Stack direction="row" spacing={2} sx={{ ml: 4 }}>
          <Button
            variant="outlined"
            startIcon={<span className="material-icons">add</span>}
            disabled={Object.keys(selectedRow).length === 0}
            onClick={(event) =>
              setModals({
                ...modals,
                dataDistribution: {
                  ...modals["dataDistribution"],
                  isVisible: true,
                },
              })
            }
          >
            Add data
          </Button>
          <Button
            variant="contained"
            startIcon={<span className="material-icons">add</span>}
            disabled={Object.keys(selectedRow).length === 0}
            onClick={(event) =>
              setModals({
                ...modals,
                trainingDistribution: {
                  ...modals["trainingDistribution"],
                  isVisible: true,
                },
              })
            }
          >
            Add training distribution
          </Button>
        </Stack>
      </Toolbar>
      <ModalDistributionForm
        isOpen={modals.trainingDistribution.isVisible}
        modalTitle={"Add environment training data distribution"}
        modalContent={"Saving environment training data distribution..."}
        modalForm={AddTrainingDataDistributionForm}
        modalButtonText={"Close"}
        initialFormValues={{
          environment_id: selectedRow.environment_id,
          data_distribution:
            Object.keys(selectedRow).length &&
            selectedRow.train_data_distribution.map((distribution) => {
              return { [Object.keys(distribution)[0]]: 0 };
            }),
          dataset_length: 0,
        }}
        headerModals={modals}
        setHeaderModalsState={setModals}
        activeHeaderModal={"trainingDistribution"}
      />
      <ModalDistributionForm
        isOpen={modals.dataDistribution.isVisible}
        modalTitle={"Add environment data distribution"}
        modalContent={"Saving environment data distribution..."}
        modalForm={DataUploadFormContainer}
        modalButtonText={"Close"}
        initialFormValues={{
          environment_id: selectedRow.environment_id,
          user: user,
          dataName: "train_data",
          labelsName: "train_labels",
          train_data: {},
          train_labels: {},
          validation_data: {},
          validation_labels: {},
          test_data: {},
          test_labels: {},
        }}
        headerModals={modals}
        setHeaderModalsState={setModals}
        activeHeaderModal={"dataDistribution"}
      />
      <Divider />
    </Box>
  );
};

export default DatasetsDataGridHeader;

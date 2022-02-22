import React from "react";
import Box from "@mui/material/Box";
import { Divider } from "@mui/material";
import Toolbar from "@mui/material/Toolbar";
import Stack from "@mui/material/Stack";
import Button from "@mui/material/Button";
import Typography from "@mui/material/Typography";
import ModalCompletedStatusForm from "./modalCompletedStatusForm";
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
      isMultipartForm: true,
    },
  });

  const [dataDistribution, setDataDistribution] = React.useState({});

  React.useEffect(() => {
    if (Object.keys(selectedRow).length > 0) {
      const dataDistributionObject = {};
      selectedRow.train_data_distribution.forEach((distribution) => {
        dataDistributionObject[Object.keys(distribution)[0]] = 0;
      });
      setDataDistribution(dataDistributionObject);
    }
  }, [selectedRow]);

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
      <ModalCompletedStatusForm
        isOpen={modals.trainingDistribution.isVisible}
        modalTitle={"Add environment training data distribution"}
        modalContent={"Saving environment training data distribution..."}
        modalForm={AddTrainingDataDistributionForm}
        modalButtonText={"Close"}
        initialFormValues={{
          user_id: user.user_id,
          environment_id: selectedRow.environment_id,
          data_distribution:
            Object.keys(selectedRow).length && dataDistribution,
          dataset_length: 0,
        }}
        headerModals={modals}
        setHeaderModalsState={setModals}
        activeHeaderModal={"trainingDistribution"}
      />
      <ModalCompletedStatusForm
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

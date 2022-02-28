import React from "react";
import Box from "@mui/material/Box";
import { Divider } from "@mui/material";
import Toolbar from "@mui/material/Toolbar";
import Stack from "@mui/material/Stack";
import Button from "@mui/material/Button";
import Typography from "@mui/material/Typography";
import Link from "next/link";
import ModalProgress from "../utils/modalProgress";
import axios from "axios";
import { getConfig } from "../../config/defaultConfig";
import ModalCompletedStatusForm from "../dataset/modalCompletedStatusForm";
import AddModelForm from "../model/addModelForm";
import ModalTrainModel from "./modalTrainModel";

const EnvironmentsDataGridHeader = ({ selectedRow }) => {
  const [progressModal, setProgressModal] = React.useState({
    isVisible: false,
    jobLink: null,
  });

  const [modals, setModals] = React.useState({
    model: {
      isVisible: false,
      url: "environmentModelAddUrl",
    },
    modelTrain: {
      isVisible: false,
    },
  });

  const deleteEnvironments = async () => {
    setProgressModal({ isVisible: true });
    try {
      const res = await axios.delete(getConfig()["environmentDeleteUrl"], {
        withCredentials: true,
        data: { environment_id: selectedRow._id },
      });
      setProgressModal({ isVisible: true, ...res.data });
    } catch (error) {
      console.log(error);
    }
  };
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
          Environments
        </Typography>
        <Stack direction="row" spacing={2} sx={{ ml: 4 }}>
          <Link href="/environment/createEnvironment">
            <Button
              variant="outlined"
              startIcon={<span className="material-icons">add</span>}
            >
              Create environment
            </Button>
          </Link>
          <Button
            variant="contained"
            startIcon={<span className="material-icons">delete</span>}
            onClick={() => deleteEnvironments()}
            disabled={Object.keys(selectedRow).length === 0}
          >
            Delete environment
          </Button>
          <Button
            variant="contained"
            startIcon={<span className="material-icons">add</span>}
            onClick={() =>
              setModals({
                ...modals,
                model: {
                  ...modals.model,
                  isVisible: true,
                },
              })
            }
            disabled={
              Object.keys(selectedRow).length === 0 ||
              (Object.keys(selectedRow).length > 0 &&
                selectedRow.status === "Training")
            }
          >
            Add model
          </Button>
          <Button
            variant="contained"
            startIcon={<span className="material-icons">fitness_center</span>}
            onClick={() =>
              setModals({
                ...modals,
                modelTrain: { isVisible: true },
              })
            }
            disabled={
              Object.keys(selectedRow).length === 0 ||
              (Object.keys(selectedRow).length > 0 &&
                selectedRow.status !== "Ready to train") ||
              (Object.keys(selectedRow).length > 0 &&
                selectedRow.status === "Training")
            }
          >
            Train model
          </Button>
        </Stack>
        <ModalProgress
          isOpen={progressModal.isVisible}
          modalButtonText={"Close"}
          modalTitle={"Environment deletion progress"}
          modalContent={"Deleting environment..."}
          jobLink={progressModal.jobLink}
        />
        <ModalCompletedStatusForm
          isOpen={modals.model.isVisible}
          modalTitle={"Add model layers"}
          modalContent={"Saving model layers..."}
          modalForm={AddModelForm}
          modalButtonText={"Close"}
          initialFormValues={{
            environment_id: selectedRow.id,
            environment_model_network_options: {
              network: [],
            },
          }}
          headerModals={modals}
          setHeaderModalsState={setModals}
          activeHeaderModal={"model"}
        />

        <ModalTrainModel
          isOpen={modals.modelTrain.isVisible}
          initialFormValues={{
            environment_id: selectedRow.id,
            training_iterations: 1,
            environment_parameters: {},
            training_options: {
              max_trials: 0,
              required_instances: 1,
            },
          }}
          headerModals={modals}
          setHeaderModalsState={setModals}
          activeHeaderModal={"modelTrain"}
        />
      </Toolbar>
      <Divider />
    </Box>
  );
};

export default EnvironmentsDataGridHeader;

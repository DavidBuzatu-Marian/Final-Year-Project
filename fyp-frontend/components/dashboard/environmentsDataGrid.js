import * as React from "react";
import { DataGrid } from "@mui/x-data-grid";
import {
  useEnvironment,
  useEnvironmentTrainingLogs,
} from "../../hooks/environment";
import { CircularProgress, Typography, Stack } from "@mui/material";
import ModalHandler from "../utils/modalHandler";
import { statuses } from "./statuses";
import moment from "moment";
import SnackbarAlert from "../alert/snackbarAlert";

const EnvironmentsDataGrid = ({ setSelectedRow }) => {
  const [environments, { loading, mutate }, environmentsError] =
    useEnvironment();
  const [trainingLogs, { loadingTrainingLogs }, trainingLogsError] =
    useEnvironmentTrainingLogs();

  const columns = [
    { field: "id", headerName: "ID", width: 220 },
    {
      field: "training_log",
      headerName: "Training logs",
      width: 250,
      renderCell: (params) => {
        return (
          <ModalHandler
            modalTitle={"Training logs"}
            modalContent={params.value}
            modalButtonText={"Open training logs"}
          />
        );
      },
    },
    {
      field: "environment_ips",
      headerName: "Environment IP addresses",
      width: 250,
      renderCell: (params) => {
        return (
          <ModalHandler
            modalTitle={"Environment instances"}
            modalContent={params.value}
            modalButtonText={"Open environment instances"}
          />
        );
      },
    },
    {
      field: "environment_options",
      headerName: "Environment options",
      renderCell: (params) => {
        return (
          <ModalHandler
            modalTitle={"Environment options"}
            modalContent={params.value}
            modalButtonText={"Open environment options"}
          />
        );
      },
      width: 250,
    },
    {
      field: "status",
      headerName: "Status",
      width: 150,
      renderCell: (params) => {
        return params.value === statuses[0] ||
          params.value === statuses[2] ||
          params.value === statuses[3] ||
          params.value === statuses[6] ? (
          <Stack direction="row">
            {" "}
            <Typography variant="p">{params.value}</Typography>
            <CircularProgress size="1rem" sx={{ ml: 1 }} />
          </Stack>
        ) : (
          <Typography variant="p">{params.value}</Typography>
        );
      },
    },
    { field: "machine_series", headerName: "Instances series", width: 200 },
    { field: "machine_type", headerName: "Instances type", width: 200 },
    {
      field: "date",
      headerName: "Date created",
      width: 200,
      valueFormatter: (params) => {
        return moment(params.value * 1000).format("YYYY-MM-DD HH:mm");
      },
    },
  ];

  return (
    <div style={{ height: 480, width: "100%" }}>
      {loading || loadingTrainingLogs ? (
        <CircularProgress sx={{ ml: 5 }} />
      ) : trainingLogsError || environmentsError ? (
        <SnackbarAlert
          message={
            "Something went wrong with the request on our part. Please try to reload"
          }
        />
      ) : (
        <DataGrid
          rows={
            environments
              ? environments.map((environment) => {
                  const environmentLog = trainingLogs
                    .filter(
                      (log) =>
                        log.environment_id === environment._id &&
                        log.user_id === environment.user_id
                    )
                    .map((log) => log.train_logs);
                  return { ...environment, training_log: environmentLog };
                })
              : []
          }
          columns={columns}
          pageSize={5}
          rowsPerPageOptions={[5]}
          onSelectionModelChange={(ids) => {
            const selectedIDs = new Set(ids);
            const selectedRowData =
              environments &&
              environments.filter((row) => selectedIDs.has(row._id));
            if (!selectedRowData || selectedRowData.length === 0) {
              setSelectedRow({});
            } else {
              setSelectedRow(...selectedRowData);
            }
          }}
        />
      )}
    </div>
  );
};

export default EnvironmentsDataGrid;

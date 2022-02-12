import * as React from 'react';
import { DataGrid, GridToolbar } from '@mui/x-data-grid';
import { useEnvironment } from '../../hooks/environment';
import { CircularProgress, Box } from '@mui/material';
import ModalHandler from '../utils/modalHandler';

const EnvironmentsDataGrid = ({ setSelectedRows }) => {
  const [environments, { loading, mutate }] = useEnvironment();
  const columns = [
    { field: 'id', headerName: 'ID', width: 220 },
    {
      field: 'environment_ips',
      headerName: 'Environment IP addresses',
      width: 250,
      renderCell: (params) => {
        return (
          <ModalHandler
            modalTitle={'Environment instances'}
            modalContent={params.value}
            modalButtonText={'Open environment instances'}
          />
        );
      },
    },
    {
      field: 'environment_options',
      headerName: 'Environment options',
      renderCell: (params) => {
        return (
          <ModalHandler
            modalTitle={'Environment options'}
            modalContent={params.value}
            modalButtonText={'Open environment options'}
          />
        );
      },
      width: 250,
    },
    {
      field: 'status',
      headerName: 'Status',
      width: 150,
    },
    { field: 'machine_series', headerName: 'Instances series', width: 200 },
    { field: 'machine_type', headerName: 'Instances type', width: 200 },
    {
      field: 'date',
      headerName: 'Date created',
      width: 200,
      valueFormatter: (params) => {
        return new Date(params.value).toLocaleString();
      },
    },
  ];

  return (
    <div style={{ height: 480, width: '100%' }}>
      {loading ? (
        <CircularProgress />
      ) : (
        <DataGrid
          rows={environments ? environments : []}
          columns={columns}
          pageSize={5}
          rowsPerPageOptions={[5]}
          onSelectionModelChange={(ids) => {
            const selectedIDs = new Set(ids);
            const selectedRowData = environments.filter((row) =>
              selectedIDs.has(row._id)
            );
            setSelectedRows(selectedRowData);
          }}
        />
      )}
    </div>
  );
};

export default EnvironmentsDataGrid;

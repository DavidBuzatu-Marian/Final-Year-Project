import * as React from 'react';
import { DataGrid, GridToolbar } from '@mui/x-data-grid';
import { useEnvironment } from '../../hooks/environment';

const EnvironmentsDataGrid = () => {
  const [environments, { loading, mutate }] = useEnvironment();

  const columns = [
    { field: 'id', headerName: 'ID', width: 90 },
    {
      field: 'environment_ips',
      headerName: 'Environment IP addresses',
      width: 250,
    },
    {
      field: 'environment_options',
      headerName: 'Environment options',
      width: 250,
    },
    {
      field: 'status',
      headerName: 'Status',
      width: 150,
    },
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
      <DataGrid
        rows={environments}
        columns={columns}
        pageSize={5}
        rowsPerPageOptions={[5]}
        checkboxSelection
      />
    </div>
  );
};

export default EnvironmentsDataGrid;

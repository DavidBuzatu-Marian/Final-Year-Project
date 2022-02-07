import React, { useEffect } from 'react';
import { DataGrid, GridToolbar } from '@mui/x-data-grid';
import {
  useDatasetDataDistribution,
  useDatasetTrainingDistribution,
} from '../../hooks/dataset';
import { CircularProgress, Box } from '@mui/material';
import ModalHandler from './modalHandler';

const DatasetsDataGrid = () => {
  const [environmentsDataDistribution, { loadingDataDistribution, mutate }] =
    useDatasetDataDistribution();
  const [
    environmentsTrainingDistribution,
    { loadingTrainingDistribution, mutate },
  ] = useDatasetTrainingDistribution();
  const [environmentsDataDistributions, setEnvironmentsDataDistributions] =
    useState([]);

  useEffect(() => {
    if (!loadingDataDistribution && !loadingTrainingDistribution) {
      const distributionMap = new Map();
      environmentsDataDistribution.map((distribution) => {
        distributionMap.set(
          distribution.user_id + distribution.environment_id,
          distribution
        );
      });
      environmentsTrainingDistribution.map((distribution) => {
        const key = distribution.user_id + distribution.environment_id;
        if (distributionMap.has(key)) {
          distributionMap.set(key, {
            ...distributionMap.get(key),
            ...distribution,
          });
        }
      });
      console.log(distributionMap);
      setEnvironmentsDataDistributions(distributionMap);
    }
  }, [environmentsDataDistribution, environmentsTrainingDistribution, loadin]);

  const columns = [
    { field: 'id', headerName: 'ID', width: 220 },
    {
      field: 'environment_id',
      headerName: 'Environment ID',
      width: 250,
    },
    {
      field: 'training_data_distribution',
      headerName: 'Environment training data distribution',
      renderCell: (params) => {
        return (
          <ModalHandler
            modalTitle={'Environment training data distribution'}
            modalContent={params.value}
            modalButtonText={'Open environment training data distribution'}
          />
        );
      },
      width: 350,
    },
    {
      field: 'training_labels_data_distribution',
      headerName: 'Environment training labels data distribution',
      renderCell: (params) => {
        return (
          <ModalHandler
            modalTitle={'Environment training labels data distribution'}
            modalContent={params.value}
            modalButtonText={
              'Open environment training labels data distribution'
            }
          />
        );
      },
      width: 350,
    },
    {
      field: 'validation_data_distribution',
      headerName: 'Environment validation data distribution',
      renderCell: (params) => {
        return (
          <ModalHandler
            modalTitle={'Environment validation data distribution'}
            modalContent={params.value}
            modalButtonText={'Open environment validation data distribution'}
          />
        );
      },
      width: 350,
    },
    {
      field: 'validation_labels_data_distribution',
      headerName: 'Environment validation labels data distribution',
      renderCell: (params) => {
        return (
          <ModalHandler
            modalTitle={'Environment validation labels data distribution'}
            modalContent={params.value}
            modalButtonText={
              'Open environment validation labels data distribution'
            }
          />
        );
      },
      width: 350,
    },
    {
      field: 'test_data_distribution',
      headerName: 'Environment test data distribution',
      renderCell: (params) => {
        return (
          <ModalHandler
            modalTitle={'Environment test data distribution'}
            modalContent={params.value}
            modalButtonText={'Open environment test data distribution'}
          />
        );
      },
      width: 350,
    },
    {
      field: 'test_labels_data_distribution',
      headerName: 'Environment test labels data distribution',
      renderCell: (params) => {
        return (
          <ModalHandler
            modalTitle={'Environment test labels data distribution'}
            modalContent={params.value}
            modalButtonText={'Open environment test labels data distribution'}
          />
        );
      },
      width: 350,
    },
    {
      field: 'distributions',
      headerName: 'Environment training data distribution split',
      renderCell: (params) => {
        return (
          <ModalHandler
            modalTitle={'Environment training data distribution split'}
            modalContent={params.value}
            modalButtonText={
              'Open environment training data distribution split'
            }
          />
        );
      },
      width: 400,
    },
  ];

  return (
    <div style={{ height: 480, width: '100%' }}>
      {loading ? (
        <CircularProgress />
      ) : (
        <DataGrid
          rows={environmentsDataDistributions}
          columns={columns}
          pageSize={5}
          rowsPerPageOptions={[5]}
          // onSelectionModelChange={(ids) => {
          //   const selectedIDs = new Set(ids);
          //   const selectedRowData = datasets.filter((row) =>
          //     selectedIDs.has(row._id)
          //   );
          // }}
        />
      )}
    </div>
  );
};

export default DatasetsDataGrid;
